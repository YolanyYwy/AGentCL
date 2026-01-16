import json
import random

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

# Select random tasks
random.seed(777)
random_task_ids = random.sample(range(114, 300), 10)
random_task_ids.sort()

print("="*80)
print("DETAILED VALIDATION OF RANDOM RETAIL TASKS")
print("="*80)

for task_id in random_task_ids:
    task = tasks[task_id]
    print(f"\n{'='*80}")
    print(f"TASK {task_id}: {task['description']['purpose']}")
    print(f"{'='*80}")

    # Find user info from authentication action
    user_info = None
    for action in task['evaluation_criteria']['actions']:
        if action['name'] == 'find_user_id_by_email':
            email = action['arguments']['email']
            print(f"\n[AUTH] Email: {email}")
            for user in db['users'].values():
                if user['email'] == email:
                    user_info = user
                    print(f"  [OK] User found: {user['name']['first_name']} {user['name']['last_name']}")
                    print(f"  User ID: {user['user_id']}")
                    break
            if not user_info:
                print(f"  [ERROR] User not found!")
        elif action['name'] == 'find_user_id_by_name_zip':
            first_name = action['arguments']['first_name']
            last_name = action['arguments']['last_name']
            zip_code = action['arguments']['zip']
            print(f"\n[AUTH] Name: {first_name} {last_name}, Zip: {zip_code}")
            for user in db['users'].values():
                if (user['name']['first_name'] == first_name and
                    user['name']['last_name'] == last_name and
                    user['address']['zip'] == zip_code):
                    user_info = user
                    print(f"  [OK] User found")
                    print(f"  User ID: {user['user_id']}")
                    break
            if not user_info:
                print(f"  [ERROR] User not found!")

    # Check order details if present
    for action in task['evaluation_criteria']['actions']:
        if action['name'] == 'get_order_details':
            order_id = action['arguments']['order_id']
            print(f"\n[ORDER] Order ID: {order_id}")
            if order_id in db['orders']:
                order = db['orders'][order_id]
                print(f"  [OK] Order exists")
                print(f"  Status: {order['status']}")
                print(f"  User ID: {order['user_id']}")
                if user_info:
                    if order['user_id'] == user_info['user_id']:
                        print(f"  [OK] Order belongs to authenticated user")
                    else:
                        print(f"  [ERROR] Order belongs to different user!")
                print(f"  Items: {len(order['items'])}")
                for i, item in enumerate(order['items'][:3]):
                    print(f"    {i+1}. {item['name']} (item_id: {item['item_id']})")
            else:
                print(f"  [ERROR] Order not found!")

    # Check specific action types
    print(f"\n[ACTIONS]")
    for action in task['evaluation_criteria']['actions']:
        action_id = action['action_id']
        action_name = action['name']
        print(f"  {action_id}: {action_name}")

        if action_name == 'cancel_pending_order':
            order_id = action['arguments']['order_id']
            reason = action['arguments']['reason']
            print(f"    -> Canceling order {order_id}, reason: {reason}")
            if order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] == 'pending':
                    print(f"    [OK] Order is pending, can be cancelled")
                else:
                    print(f"    [ERROR] Order status is {order['status']}, not pending!")

        elif action_name == 'return_delivered_order_items':
            order_id = action['arguments']['order_id']
            item_ids = action['arguments']['item_ids']
            print(f"    -> Returning {len(item_ids)} items from order {order_id}")
            if order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] == 'delivered':
                    print(f"    [OK] Order is delivered, can be returned")
                else:
                    print(f"    [ERROR] Order status is {order['status']}, not delivered!")

        elif action_name == 'exchange_delivered_order_items':
            order_id = action['arguments']['order_id']
            old_item_ids = action['arguments']['old_item_ids']
            new_item_ids = action['arguments']['new_item_ids']
            print(f"    -> Exchanging {len(old_item_ids)} items")
            if order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] == 'delivered':
                    print(f"    [OK] Order is delivered, can be exchanged")
                    # Verify items belong to same product
                    for old_id, new_id in zip(old_item_ids, new_item_ids):
                        old_item = None
                        for item in order['items']:
                            if item['item_id'] == old_id:
                                old_item = item
                                break
                        if old_item:
                            # Find new item in products
                            for product in db['products'].values():
                                if old_item['product_id'] == product['product_id']:
                                    if new_id in product['variants']:
                                        print(f"    [OK] Exchange within same product")
                                    break
                else:
                    print(f"    [ERROR] Order status is {order['status']}, not delivered!")

        elif action_name == 'modify_pending_order_address':
            order_id = action['arguments']['order_id']
            if order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] == 'pending':
                    print(f"    [OK] Order is pending, address can be modified")
                else:
                    print(f"    [ERROR] Order status is {order['status']}, not pending!")

        elif action_name == 'modify_pending_order_payment':
            order_id = action['arguments']['order_id']
            if order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] == 'pending':
                    print(f"    [OK] Order is pending, payment can be modified")
                else:
                    print(f"    [ERROR] Order status is {order['status']}, not pending!")

    print(f"\n>>> TASK {task_id} VALIDATION COMPLETE <<<")

print(f"\n{'='*80}")
print("DETAILED VALIDATION COMPLETE")
print(f"{'='*80}")
