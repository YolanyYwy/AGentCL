#!/usr/bin/env python3
"""
Accelerate 多卡训练测试脚本
测试 Accelerate 是否正确配置并能正常运行
"""

import torch
import torch.nn as nn
from accelerate import Accelerator
import time


def test_accelerate_basic():
    """测试 1: Accelerate 基础功能"""
    print("\n" + "="*80)
    print("测试 1: Accelerate 基础功能")
    print("="*80)

    try:
        accelerator = Accelerator()

        print(f"✅ Accelerator 初始化成功")
        print(f"  - 进程数量: {accelerator.num_processes}")
        print(f"  - 当前进程: {accelerator.process_index}")
        print(f"  - 设备: {accelerator.device}")
        print(f"  - 是否主进程: {accelerator.is_main_process}")
        print(f"  - 混合精度: {accelerator.mixed_precision}")

        return True
    except Exception as e:
        print(f"❌ Accelerator 初始化失败: {e}")
        return False


def test_model_distribution():
    """测试 2: 模型分布"""
    print("\n" + "="*80)
    print("测试 2: 模型分布到多 GPU")
    print("="*80)

    try:
        accelerator = Accelerator()

        # 创建简单模型
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Prepare with Accelerator
        model, optimizer = accelerator.prepare(model, optimizer)

        if accelerator.is_main_process:
            print(f"✅ 模型已分布到所有 GPU")
            print(f"  - 模型设备: {next(model.parameters()).device}")

        return True
    except Exception as e:
        print(f"❌ 模型分布失败: {e}")
        return False


def test_gradient_sync():
    """测试 3: 梯度同步"""
    print("\n" + "="*80)
    print("测试 3: 梯度同步")
    print("="*80)

    try:
        accelerator = Accelerator()

        # 创建模型
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model, optimizer = accelerator.prepare(model, optimizer)

        # 创建不同的数据（每个 GPU 不同）
        x = torch.randn(4, 10, device=accelerator.device) * (accelerator.process_index + 1)
        y = torch.randn(4, 1, device=accelerator.device)

        # 前向传播
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        # 反向传播
        accelerator.backward(loss)

        # 获取梯度
        grad_before = model.weight.grad.clone()

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()

        # 等待所有进程
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"✅ 梯度同步成功")
            print(f"  - Loss: {loss.item():.4f}")
            print(f"  - 梯度范数: {grad_before.norm().item():.4f}")

        return True
    except Exception as e:
        print(f"❌ 梯度同步失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_distribution():
    """测试 4: 数据分布"""
    print("\n" + "="*80)
    print("测试 4: 数据分布")
    print("="*80)

    try:
        accelerator = Accelerator()

        # 创建数据
        data = list(range(100))

        # 每个进程处理不同的数据
        per_device = len(data) // accelerator.num_processes
        start_idx = accelerator.process_index * per_device
        end_idx = start_idx + per_device

        my_data = data[start_idx:end_idx]

        print(f"[进程 {accelerator.process_index}] 处理数据: {len(my_data)} 个样本")
        print(f"[进程 {accelerator.process_index}] 数据范围: {my_data[0]} - {my_data[-1]}")

        # 等待所有进程
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"✅ 数据分布成功")

        return True
    except Exception as e:
        print(f"❌ 数据分布失败: {e}")
        return False


def test_gather_operation():
    """测试 5: Gather 操作"""
    print("\n" + "="*80)
    print("测试 5: Gather 操作（收集所有 GPU 的数据）")
    print("="*80)

    try:
        accelerator = Accelerator()

        # 每个进程创建不同的数据
        local_data = torch.tensor([accelerator.process_index], device=accelerator.device)

        # Gather 到所有进程
        gathered_data = accelerator.gather(local_data)

        if accelerator.is_main_process:
            print(f"✅ Gather 操作成功")
            print(f"  - 收集的数据: {gathered_data.cpu().tolist()}")

        return True
    except Exception as e:
        print(f"❌ Gather 操作失败: {e}")
        return False


def test_save_load():
    """测试 6: 模型保存和加载"""
    print("\n" + "="*80)
    print("测试 6: 模型保存和加载")
    print("="*80)

    try:
        accelerator = Accelerator()

        # 创建模型
        model = nn.Linear(10, 5)
        model, = accelerator.prepare(model)

        # 保存模型（只在主进程）
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), "/tmp/test_model.pt")
            print(f"✅ 模型保存成功: /tmp/test_model.pt")

        # 等待保存完成
        accelerator.wait_for_everyone()

        # 加载模型
        new_model = nn.Linear(10, 5)
        new_model.load_state_dict(torch.load("/tmp/test_model.pt"))

        if accelerator.is_main_process:
            print(f"✅ 模型加载成功")

        return True
    except Exception as e:
        print(f"❌ 模型保存/加载失败: {e}")
        return False


def test_training_loop():
    """测试 7: 完整训练循环"""
    print("\n" + "="*80)
    print("测试 7: 完整训练循环")
    print("="*80)

    try:
        accelerator = Accelerator()

        # 创建模型和优化器
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model, optimizer = accelerator.prepare(model, optimizer)

        # 训练循环
        num_steps = 10
        start_time = time.time()

        for step in range(num_steps):
            # 创建随机数据
            x = torch.randn(8, 20, device=accelerator.device)
            y = torch.randint(0, 10, (8,), device=accelerator.device)

            # 前向传播
            output = model(x)
            loss = nn.functional.cross_entropy(output, y)

            # 反向传播
            accelerator.backward(loss)

            # 梯度裁剪
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            # 优化器步骤
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process and step % 5 == 0:
                print(f"  Step {step}/{num_steps}, Loss: {loss.item():.4f}")

        elapsed = time.time() - start_time

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print(f"✅ 训练循环成功")
            print(f"  - 总步数: {num_steps}")
            print(f"  - 耗时: {elapsed:.2f}s")
            print(f"  - 速度: {num_steps/elapsed:.2f} steps/s")

        return True
    except Exception as e:
        print(f"❌ 训练循环失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mixed_precision():
    """测试 8: 混合精度训练"""
    print("\n" + "="*80)
    print("测试 8: 混合精度训练")
    print("="*80)

    try:
        accelerator = Accelerator(mixed_precision="bf16")

        model = nn.Linear(100, 100)
        optimizer = torch.optim.Adam(model.parameters())

        model, optimizer = accelerator.prepare(model, optimizer)

        # 训练一步
        x = torch.randn(4, 100, device=accelerator.device)
        y = torch.randn(4, 100, device=accelerator.device)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.is_main_process:
            print(f"✅ 混合精度训练成功")
            print(f"  - 混合精度类型: {accelerator.mixed_precision}")
            print(f"  - 参数 dtype: {next(model.parameters()).dtype}")

        return True
    except Exception as e:
        print(f"❌ 混合精度训练失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Accelerate 多卡训练测试套件")
    print("="*80)

    # 检查 CUDA
    print(f"\n🔍 环境检查:")
    print(f"  - PyTorch 版本: {torch.__version__}")
    print(f"  - CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA 版本: {torch.version.cuda}")
        print(f"  - GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # 运行测试
    tests = [
        ("基础功能", test_accelerate_basic),
        ("模型分布", test_model_distribution),
        ("梯度同步", test_gradient_sync),
        ("数据分布", test_data_distribution),
        ("Gather 操作", test_gather_operation),
        ("保存/加载", test_save_load),
        ("训练循环", test_training_loop),
        ("混合精度", test_mixed_precision),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 异常: {e}")
            results.append((name, False))

    # 打印总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！Accelerate 多卡训练配置正确！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查配置")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
