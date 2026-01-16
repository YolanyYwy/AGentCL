"""
检查airline和retail域的所有生成任务
验证内容：
1. 数据库一致性（用户、订单、航班、产品等是否存在）
2. 工具调用顺序和参数正确性
3. 业务逻辑合规性（状态检查、权限检查等）
"""

import json
from datetime import datetime
from typing import Dict, List, Tuple

class TaskValidator:
    def __init__(self, domain: str):
        self.domain = domain
        self.db = None
        self.tasks = None
        self.errors = []
        self.warnings = []

    def load_data(self):
        """加载数据库和任务数据"""
        db_path = rf'D:\Desktop\work\tau2-bench\data\tau2\domains\{self.domain}\db.json'
        tasks_path = rf'D:\Desktop\work\tau2-bench\data\tau2\domains\{self.domain}\tasks.json'

        with open(db_path, 'r', encoding='utf-8') as f:
            self.db = json.load(f)

        with open(tasks_path, 'r', encoding='utf-8') as f:
            self.tasks = json.load(f)

    def add_error(self, task_id: str, message: str):
        """添加错误"""
        self.errors.append(f"Task {task_id}: {message}")

    def add_warning(self, task_id: str, message: str):
        """添加警告"""
        self.warnings.append(f"Task {task_id}: {message}")


class AirlineValidator(TaskValidator):
    def __init__(self):
        super().__init__('airline')
        self.current_time = datetime(2024, 5, 15, 15, 0, 0)

    def validate_all_tasks(self, start_id: int = 50):
        """验证所有新生成的任务"""
        print(f"\n{'='*80}")
        print(f"验证 AIRLINE 域任务 (ID {start_id}-299)")
        print(f"{'='*80}")

        for task in self.tasks[start_id:]:
            task_id = task['id']
            self.validate_task(task)

        self.print_results()

    def validate_task(self, task: Dict):
        """验证单个任务"""
        task_id = task['id']
        actions = task['evaluation_criteria']['actions']

        # 1. 验证action_id格式
        for i, action in enumerate(actions):
            expected_id = f"{task_id}_{i}"
            if action['action_id'] != expected_id:
                self.add_error(task_id, f"action_id错误: 期望 {expected_id}, 实际 {action['action_id']}")

        # 2. 验证工具调用顺序和参数
        user_id = None
        reservation_id = None

        for action in actions:
            action_name = action['name']
            args = action['arguments']

            if action_name == 'get_user_details':
                user_id = args.get('user_id')
                if not self.validate_user_exists(task_id, user_id):
                    continue

            elif action_name == 'get_reservation_details':
                reservation_id = args.get('reservation_id')
                if not self.validate_reservation_exists(task_id, reservation_id):
                    continue

                # 验证预订属于用户
                if user_id:
                    reservation = self.db['reservations'].get(reservation_id)
                    if reservation and reservation['user_id'] != user_id:
                        self.add_error(task_id, f"预订 {reservation_id} 不属于用户 {user_id}")

            elif action_name == 'search_direct_flight':
                origin = args.get('origin')
                destination = args.get('destination')
                if not self.validate_flight_route_exists(task_id, origin, destination):
                    self.add_warning(task_id, f"航线 {origin}->{destination} 可能不存在")

            elif action_name == 'cancel_reservation':
                if not reservation_id:
                    self.add_error(task_id, "调用cancel_reservation前未获取reservation_id")
                    continue

                reservation = self.db['reservations'].get(reservation_id)
                if reservation:
                    # 检查取消资格
                    can_cancel = self.check_cancellation_eligibility(reservation)
                    if not can_cancel:
                        self.add_warning(task_id, f"预订 {reservation_id} 可能不符合取消条件")

            elif action_name == 'update_reservation_baggages':
                if not reservation_id:
                    self.add_error(task_id, "调用update_reservation_baggages前未获取reservation_id")
                    continue

                new_total = args.get('new_total_baggages')
                reservation = self.db['reservations'].get(reservation_id)
                if reservation and new_total < reservation['total_baggages']:
                    self.add_error(task_id, "不能减少行李数量，只能增加")

            elif action_name == 'update_reservation_flights':
                if not reservation_id:
                    self.add_error(task_id, "调用update_reservation_flights前未获取reservation_id")
                    continue

                reservation = self.db['reservations'].get(reservation_id)
                if reservation and reservation['cabin'] == 'basic_economy':
                    self.add_error(task_id, "基础经济舱不能修改航班")

            elif action_name == 'send_certificate':
                if not user_id:
                    self.add_error(task_id, "调用send_certificate前未获取user_id")

                amount = args.get('amount')
                if amount and amount <= 0:
                    self.add_error(task_id, f"补偿金额必须大于0: {amount}")

    def validate_user_exists(self, task_id: str, user_id: str) -> bool:
        """验证用户是否存在"""
        if user_id not in self.db['users']:
            self.add_error(task_id, f"用户 {user_id} 不存在于数据库")
            return False
        return True

    def validate_reservation_exists(self, task_id: str, reservation_id: str) -> bool:
        """验证预订是否存在"""
        if reservation_id not in self.db['reservations']:
            self.add_error(task_id, f"预订 {reservation_id} 不存在于数据库")
            return False
        return True

    def validate_flight_route_exists(self, task_id: str, origin: str, destination: str) -> bool:
        """验证航线是否存在"""
        for flight in self.db['flights'].values():
            if flight['origin'] == origin and flight['destination'] == destination:
                return True
        return False

    def check_cancellation_eligibility(self, reservation: Dict) -> bool:
        """检查是否符合取消条件"""
        # 检查是否在24小时内
        created_at = datetime.fromisoformat(reservation['created_at'].replace('T', ' '))
        hours_since = (self.current_time - created_at).total_seconds() / 3600
        within_24h = hours_since < 24

        # 检查其他条件
        is_business = reservation['cabin'] == 'business'
        has_insurance = reservation.get('insurance', 'no') == 'yes'

        return within_24h or is_business or has_insurance

    def print_results(self):
        """打印验证结果"""
        print(f"\n验证完成!")
        print(f"  错误数: {len(self.errors)}")
        print(f"  警告数: {len(self.warnings)}")

        if self.errors:
            print(f"\n错误列表 (前20个):")
            for error in self.errors[:20]:
                print(f"  [ERROR] {error}")

        if self.warnings:
            print(f"\n警告列表 (前10个):")
            for warning in self.warnings[:10]:
                print(f"  [WARNING] {warning}")

        if not self.errors and not self.warnings:
            print(f"\n[OK] 所有验证通过!")


class RetailValidator(TaskValidator):
    def __init__(self):
        super().__init__('retail')

    def validate_all_tasks(self, start_id: int = 114):
        """验证所有新生成的任务"""
        print(f"\n{'='*80}")
        print(f"验证 RETAIL 域任务 (ID {start_id}-299)")
        print(f"{'='*80}")

        for task in self.tasks[start_id:]:
            task_id = task['id']
            self.validate_task(task)

        self.print_results()

    def validate_task(self, task: Dict):
        """验证单个任务"""
        task_id = task['id']
        actions = task['evaluation_criteria']['actions']

        # 1. 验证action_id格式
        for i, action in enumerate(actions):
            expected_id = f"{task_id}_{i}"
            if action['action_id'] != expected_id:
                self.add_error(task_id, f"action_id错误: 期望 {expected_id}, 实际 {action['action_id']}")

        # 2. 验证工具调用顺序和参数
        user_id = None
        order_id = None
        authenticated = False

        for action in actions:
            action_name = action['name']
            args = action['arguments']

            # 验证认证步骤
            if action_name == 'find_user_id_by_email':
                email = args.get('email')
                if self.validate_user_by_email(task_id, email):
                    authenticated = True
                    # 获取user_id
                    for user in self.db['users'].values():
                        if user['email'] == email:
                            user_id = user['user_id']
                            break

            elif action_name == 'find_user_id_by_name_zip':
                first_name = args.get('first_name')
                last_name = args.get('last_name')
                zip_code = args.get('zip')
                if self.validate_user_by_name_zip(task_id, first_name, last_name, zip_code):
                    authenticated = True
                    # 获取user_id
                    for user in self.db['users'].values():
                        if (user['name']['first_name'] == first_name and
                            user['name']['last_name'] == last_name and
                            user['address']['zip'] == zip_code):
                            user_id = user['user_id']
                            break

            elif action_name == 'get_user_details':
                user_id = args.get('user_id')
                if not self.validate_user_exists(task_id, user_id):
                    continue

            elif action_name == 'get_order_details':
                order_id = args.get('order_id')
                if not self.validate_order_exists(task_id, order_id):
                    continue

                # 验证订单属于用户
                if user_id:
                    order = self.db['orders'].get(order_id)
                    if order and order['user_id'] != user_id:
                        self.add_error(task_id, f"订单 {order_id} 不属于用户 {user_id}")

            elif action_name == 'get_product_details':
                product_id = args.get('product_id')
                if not self.validate_product_exists(task_id, product_id):
                    continue

            # 验证需要认证的操作
            elif action_name in ['cancel_pending_order', 'modify_pending_order_address',
                                'modify_pending_order_payment', 'modify_pending_order_items',
                                'return_delivered_order_items', 'exchange_delivered_order_items',
                                'modify_user_address']:
                if not authenticated:
                    self.add_error(task_id, f"调用 {action_name} 前未进行用户认证")

            # 验证订单状态相关操作
            if action_name == 'cancel_pending_order':
                if not order_id:
                    self.add_error(task_id, "调用cancel_pending_order前未获取order_id")
                    continue

                order = self.db['orders'].get(order_id)
                if order and order['status'] != 'pending':
                    self.add_error(task_id, f"订单 {order_id} 状态为 {order['status']}, 不是pending，不能取消")

                # 验证取消原因
                reason = args.get('reason')
                if reason not in ['no longer needed', 'ordered by mistake']:
                    self.add_error(task_id, f"取消原因不合法: {reason}")

            elif action_name in ['modify_pending_order_address', 'modify_pending_order_payment',
                                'modify_pending_order_items']:
                if not order_id:
                    self.add_error(task_id, f"调用{action_name}前未获取order_id")
                    continue

                order = self.db['orders'].get(order_id)
                if order and order['status'] != 'pending':
                    self.add_error(task_id, f"订单 {order_id} 状态为 {order['status']}, 不是pending，不能修改")

            elif action_name in ['return_delivered_order_items', 'exchange_delivered_order_items']:
                if not order_id:
                    self.add_error(task_id, f"调用{action_name}前未获取order_id")
                    continue

                order = self.db['orders'].get(order_id)
                if order and order['status'] != 'delivered':
                    self.add_error(task_id, f"订单 {order_id} 状态为 {order['status']}, 不是delivered，不能退换货")

                # 验证item_ids是否属于订单
                if action_name == 'return_delivered_order_items':
                    item_ids = args.get('item_ids', [])
                    if order:
                        order_item_ids = [item['item_id'] for item in order['items']]
                        for item_id in item_ids:
                            if item_id not in order_item_ids:
                                self.add_error(task_id, f"item {item_id} 不属于订单 {order_id}")

                elif action_name == 'exchange_delivered_order_items':
                    old_item_ids = args.get('old_item_ids', [])
                    new_item_ids = args.get('new_item_ids', [])

                    if len(old_item_ids) != len(new_item_ids):
                        self.add_error(task_id, "交换的旧物品和新物品数量不匹配")

                    if order:
                        # 验证旧物品属于订单
                        order_item_ids = [item['item_id'] for item in order['items']]
                        for item_id in old_item_ids:
                            if item_id not in order_item_ids:
                                self.add_error(task_id, f"item {item_id} 不属于订单 {order_id}")

                        # 验证新旧物品属于同一产品
                        for old_id, new_id in zip(old_item_ids, new_item_ids):
                            old_item = None
                            for item in order['items']:
                                if item['item_id'] == old_id:
                                    old_item = item
                                    break

                            if old_item:
                                product = self.db['products'].get(old_item['product_id'])
                                if product and new_id not in product['variants']:
                                    self.add_error(task_id, f"新物品 {new_id} 不属于产品 {old_item['product_id']}")

    def validate_user_exists(self, task_id: str, user_id: str) -> bool:
        """验证用户是否存在"""
        if user_id not in self.db['users']:
            self.add_error(task_id, f"用户 {user_id} 不存在于数据库")
            return False
        return True

    def validate_user_by_email(self, task_id: str, email: str) -> bool:
        """通过邮箱验证用户"""
        for user in self.db['users'].values():
            if user['email'] == email:
                return True
        self.add_error(task_id, f"邮箱 {email} 不存在于数据库")
        return False

    def validate_user_by_name_zip(self, task_id: str, first_name: str,
                                  last_name: str, zip_code: str) -> bool:
        """通过姓名和邮编验证用户"""
        for user in self.db['users'].values():
            if (user['name']['first_name'] == first_name and
                user['name']['last_name'] == last_name and
                user['address']['zip'] == zip_code):
                return True
        self.add_error(task_id, f"用户 {first_name} {last_name} (zip: {zip_code}) 不存在于数据库")
        return False

    def validate_order_exists(self, task_id: str, order_id: str) -> bool:
        """验证订单是否存在"""
        if order_id not in self.db['orders']:
            self.add_error(task_id, f"订单 {order_id} 不存在于数据库")
            return False
        return True

    def validate_product_exists(self, task_id: str, product_id: str) -> bool:
        """验证产品是否存在"""
        if product_id not in self.db['products']:
            self.add_error(task_id, f"产品 {product_id} 不存在于数据库")
            return False
        return True

    def print_results(self):
        """打印验证结果"""
        print(f"\n验证完成!")
        print(f"  错误数: {len(self.errors)}")
        print(f"  警告数: {len(self.warnings)}")

        if self.errors:
            print(f"\n错误列表 (前20个):")
            for error in self.errors[:20]:
                print(f"  [ERROR] {error}")

        if self.warnings:
            print(f"\n警告列表 (前10个):")
            for warning in self.warnings[:10]:
                print(f"  [WARNING] {warning}")

        if not self.errors and not self.warnings:
            print(f"\n[OK] 所有验证通过!")


def main():
    """主函数"""
    print("="*80)
    print("全面验证 - Airline 和 Retail 域任务")
    print("="*80)

    # 验证Airline域
    print("\n[1/2] 验证 AIRLINE 域...")
    airline_validator = AirlineValidator()
    airline_validator.load_data()
    airline_validator.validate_all_tasks(start_id=50)

    # 验证Retail域
    print("\n[2/2] 验证 RETAIL 域...")
    retail_validator = RetailValidator()
    retail_validator.load_data()
    retail_validator.validate_all_tasks(start_id=114)

    # 总结
    print("\n" + "="*80)
    print("总体验证结果")
    print("="*80)

    total_errors = len(airline_validator.errors) + len(retail_validator.errors)
    total_warnings = len(airline_validator.warnings) + len(retail_validator.warnings)

    print(f"\nAirline域:")
    print(f"  错误: {len(airline_validator.errors)}")
    print(f"  警告: {len(airline_validator.warnings)}")

    print(f"\nRetail域:")
    print(f"  错误: {len(retail_validator.errors)}")
    print(f"  警告: {len(retail_validator.warnings)}")

    print(f"\n总计:")
    print(f"  总错误数: {total_errors}")
    print(f"  总警告数: {total_warnings}")

    if total_errors == 0 and total_warnings == 0:
        print(f"\n>>> 恭喜! 所有任务验证通过! <<<")
    elif total_errors == 0:
        print(f"\n[OK] 没有错误，但有 {total_warnings} 个警告需要注意")
    else:
        print(f"\n[FAIL] 发现 {total_errors} 个错误需要修复")

    print("="*80)


if __name__ == "__main__":
    main()
