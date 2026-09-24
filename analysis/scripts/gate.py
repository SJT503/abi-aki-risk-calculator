#!/usr/bin/env python3
"""
gate.py — 自动门禁模块 v2
新增: time_estimate_gate() — 费米估算门禁, 跑 1 单位实测时间, 估算总量, 超阈值报警/阻止
用法: 在任何脚本的第一行写:
    from gate import gate, gpu_gate, time_estimate_gate
    gpu_gate()
    gate(__file__)
    time_estimate_gate(sample_fn, total_units=3800, threshold_min=60, device="cpu")
"""
import os, sys, time, functools

GATE_FILE = ".gate_passed"
GATE_TTL = 1800  # 30 分钟过期

def gate(script_path=None, required_checks=None):
    """
    主入口: 验证 gate 标记存在且未过期
    如果提供了 required_checks, 也一并验证
    """
    gate_file = os.path.join(os.path.dirname(os.path.abspath(script_path or ".")), GATE_FILE)

    if not os.path.exists(gate_file):
        print(f"❌ GATE BLOCKED: {GATE_FILE} 不存在")
        print(f"   先运行 preflight 脚本: python preflight_check.py --task <任务名>")
        sys.exit(1)

    age = time.time() - os.path.getmtime(gate_file)
    if age > GATE_TTL:
        print(f"❌ GATE EXPIRED: 标记已过期 ({age/60:.0f} 分钟前)")
        print(f"   重新运行 preflight")
        sys.exit(1)

    # 额外检查
    if required_checks:
        for name, fn in required_checks.items():
            if not fn():
                print(f"❌ GATE CHECK FAILED: {name}")
                sys.exit(1)

    print(f"✅ GATE PASSED ({age/60:.0f}分钟前验证)")


def pass_gate(directory="."):
    """preflight 脚本调用: 写入标记"""
    with open(os.path.join(directory, GATE_FILE), "w") as f:
        f.write(str(time.time()))
    print(f"✅ Gate passed: {GATE_FILE} 已写入")


def gpu_gate():
    """GPU 专用检查: 不在 GPU 上就立刻退出"""
    import torch
    if not torch.cuda.is_available():
        print("❌ GPU GATE: torch.cuda.is_available() = False")
        print("   如果任务需要 GPU, 检查:")
        print("   - Kaggle: kernel-metadata.json 是否有 machine_shape 字段")
        print("   - 本地: nvidia-smi 检查驱动")
        sys.exit(1)
    print(f"✅ GPU GATE: {torch.cuda.get_device_name(0)}")


def canary_gate(fn, description="canary", min_output_size=1):
    """Canary 专用: 跑一个最小单元, 验证输出"""
    try:
        result = fn()
        if result is None or (hasattr(result, '__len__') and len(result) < min_output_size):
            print(f"❌ CANARY GATE: {description} 输出为空或过小")
            sys.exit(1)
        print(f"✅ CANARY GATE: {description} 通过")
        return result
    except Exception as e:
        print(f"❌ CANARY GATE: {description} 异常: {e}")
        sys.exit(1)


def retry_limit(max_retries=2):
    """装饰器: 限制重试次数, 超限强制退出"""
    def decorator(fn):
        counter = {'n': 0}
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            counter['n'] += 1
            if counter['n'] > max_retries:
                print(f"❌ RETRY LIMIT: {fn.__name__} 已重试 {counter['n']-1} 次 (上限 {max_retries})")
                print("   → 停止重试, 换策略: 搜索已知Issue / 换方法 / 问PI")
                sys.exit(1)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def memory_estimate_gate(batch_size, seq_len, heads, layers,
                         model_static_gb=None, model_params_m=None,
                         gpu_total_gb=14.5, safety=0.85,
                         probe_fn=None, probe_batch=1,
                         mode="two_stage"):
    """
    GPU 显存费米估算门禁 (2026-09-19, Kaggle v8/v9 OOM 教训的代码化)

    两段式:
      Stage 1 公式粗筛 (数量级): 参数内存 + 注意力激活上界
      Stage 2 探针精测 (可选但强烈推荐): 跑 probe_batch 实测
             torch.cuda.max_memory_allocated, 线性外推到 batch_size

    教训校准 (Geneformer 104M, T4, fp32):
      - v8 batch=64 公式就超 → OOM
      - v9 batch=24 线性外推超 → OOM
      - v10 batch=8 探针实测 7.25/14.56GiB → 成功
      纯公式(系数1.0)对 batch8 也报超 → 公式只能粗筛, 批量缩放必须探针外推

    参数:
        batch_size:     目标 batch
        seq_len:        序列长度 (Geneformer 2048/4096)
        heads:          注意力头数
        layers:         层数
        model_static_gb: 实测模型静态占用 (权重+优化器+碎片; Geneformer≈5.6)
        model_params_m:  参数量(百万), 用于未给 static 时的粗估 (fp32: ×4MB/M)
        gpu_total_gb:   GPU 总显存 (T4×2 可用 14.56; 默认 14.5)
        safety:         安全系数 (默认 0.85, 留碎片/峰值余量)
        probe_fn:       可调用, 参数为 batch, 跑 1-2 step 训练/推理 (需真实前向)
        probe_batch:    探针用的 batch (默认 1)
        mode:           "two_stage"(默认) / "formula"(仅公式) / "probe"(仅探针)

    返回:
        通过 → (est_total_gb, avail_gb)
        超限 → sys.exit(1) 并打印建议 batch 上界
    """
    avail_gb = gpu_total_gb * safety

    # ---------- Stage 1: 公式粗筛 ----------
    if model_static_gb is None:
        if model_params_m is None:
            print("❌ MEMORY GATE: 需提供 model_static_gb 或 model_params_m")
            sys.exit(1)
        model_static_gb = model_params_m * 4 / 1000  # fp32: 4 bytes/param

    attn_upper_gb = batch_size * heads * seq_len * seq_len * 4 / 1e9 * 2 * layers
    formula_total = model_static_gb + attn_upper_gb
    print(f"🧠 显存粗筛 (公式上界):")
    print(f"   模型静态: {model_static_gb:.2f} GB")
    print(f"   注意力上界 (b{batch_size}·h{heads}·s{seq_len}·L{layers}全驻留): {attn_upper_gb:.1f} GB")
    print(f"   上界合计: {formula_total:.1f} GB vs 可用 {avail_gb:.1f} GB")

    if mode == "formula":
        if formula_total > avail_gb * 3:
            _mem_block(batch_size, formula_total, avail_gb)
        else:
            print(f"✅ MEMORY GATE (formula): 上界在 3× 可用内, 视为可行 (精确判断需探针)")
            return formula_total, avail_gb

    if formula_total > avail_gb * 5:
        # 差 5 倍以上, 连激活优化都救不回来, 直接 block
        print(f"   上界超可用 5×, 任何实现都救不回 → 直接拦截")
        _mem_block(batch_size, formula_total, avail_gb)

    # ---------- Stage 2: 探针精测 ----------
    if probe_fn is None:
        print(f"⚠️ 未提供 probe_fn, 仅完成粗筛 (上界 {formula_total:.1f} GB, 激活优化后大概率安全)")
        print(f"   强烈建议 probe_fn=lambda b: 训练1step, 精确外推 batch 上限")
        return formula_total, avail_gb

    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_probe = probe_fn(probe_batch)
    peak_probe = torch.cuda.max_memory_allocated() / 1e9
    static_probe = peak_probe - (probe_batch * heads * seq_len * seq_len * 4 / 1e9 * 2 * layers) * 0.13  # 剥离探针激活(校准系数)
    act_probe = peak_probe - static_probe
    # 线性外推激活
    est_act = act_probe * (batch_size / probe_batch)
    est_total = static_probe + est_act
    print(f"🧠 探针精测 (batch={probe_batch}):")
    print(f"   实测峰值: {peak_probe:.2f} GB (静态≈{static_probe:.2f}, 激活≈{act_probe:.2f})")
    print(f"   线性外推 batch={batch_size}: {est_total:.2f} GB vs 可用 {avail_gb:.1f} GB")

    if est_total > avail_gb:
        _mem_block(batch_size, est_total, avail_gb, static=static_probe, act_per_batch=act_probe/probe_batch)

    print(f"✅ MEMORY GATE PASSED: {est_total:.2f}/{avail_gb:.1f} GB")
    return est_total, avail_gb


def _mem_block(batch_size, est, avail, static=None, act_per_batch=None):
    print(f"❌ MEMORY GATE BLOCKED: 预计 {est:.1f} GB > 可用 {avail:.1f} GB")
    if act_per_batch and static is not None:
        safe_batch = max(1, int((avail - static) / act_per_batch))
        print(f"   💡 建议缩到 batch ≤ {safe_batch} (静态{static:.1f}GB + 每batch激活{act_per_batch:.2f}GB)")
    print(f"   或: 缩短 seq_len / 用梯度检查点 / fp16 / 换更大显存GPU")
    sys.exit(1)


def time_estimate_gate(sample_fn, total_units, threshold_min=60, device="auto", unit_name="unit"):
    """
    费米估算门禁: 跑 1 单位实测时间 → 估算总量 → 超阈值报警或阻止

    参数:
        sample_fn:      可调用对象, 执行 1 个工作单元 (如: 1 个 batch 的前向推理)
        total_units:    总工作单元数 (如: 200 批 × 19 KO = 3800)
        threshold_min:  估算总时间超过此分钟数 → 阻止 (默认 60 分钟)
        device:         "auto" 自动检测 / "cpu" / "cuda"
        unit_name:      单位名 (用于日志, 如 "batch", "样本", "KO")

    返回:
        通过 → 不返回 (继续执行)
        超时 → 打印警告并 sys.exit(1)
    """
    import time as _time

    # 设备检测
    if device == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            dev = "cpu"
    else:
        dev = device

    # 跑 1 单位, 实测时间
    t0 = _time.perf_counter()
    sample_fn()
    t_per_unit = _time.perf_counter() - t0

    # 估算总量
    total_sec = t_per_unit * total_units
    total_min = total_sec / 60
    total_hr = total_min / 60

    # 判断
    print(f"⏱️  费米估算 ({dev}):")
    print(f"   1 {unit_name} = {t_per_unit:.2f}s (实测)")
    print(f"   {total_units:,} {unit_name}s × {t_per_unit:.2f}s = {total_min:.0f} 分钟 ({total_hr:.1f} 小时)")

    if total_min > threshold_min:
        print(f"❌ TIME GATE BLOCKED: 预计 {total_hr:.1f} 小时 > 阈值 {threshold_min} 分钟")
        if dev == "cpu":
            print(f"   💡 当前在 CPU 上. 这个工作量的合理选择:")
            print(f"      - Kaggle T4 GPU (kaggle kernels push, machine_shape=NvidiaTeslaT4)")
            print(f"      - 或减少 total_units / 降低模型复杂度")
        sys.exit(1)

    print(f"✅ TIME GATE PASSED: {total_min:.0f} 分钟 < {threshold_min} 分钟阈值")
    return total_min
