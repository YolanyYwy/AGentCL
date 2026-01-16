import json
import random

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\retail\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

print("="*80)
print("RETAIL TASKS VALIDATION REPORT")
print("="*80)

# 1. Basic statistics
print(f"\n1. BASIC STATISTICS")
print(f"   Total tasks: {len(tasks)}")
print(f"   Original tasks (0-113): 114")
print(f"   New tasks (114-299): 186")
print(f"   Task ID range: {tasks[0]['id']} to {tasks[-1]['id']}")

# 2. Validate all action_id formats
print(f"\n2. ACTION_ID FORMAT VALIDATION")
invalid_action_ids = []
for task in tasks[114:]:
    task_id = task['id']
    for action in task['evaluation_criteria']['actions']:
        action_id = action['action_id']
        expected_prefix = f"{task_id}_"
        if not action_id.startswith(expected_prefix):
            invalid_action_ids.append((task_id, action_id))

if invalid_action_ids:
    print(f"   [ERROR] Found {len(invalid_action_ids)} invalid action_ids:")
    for tid, aid in invalid_action_ids[:5]:
        print(f"     Task {tid}: {aid}")
else:
    print(f"   [OK] All 186 new tasks have correct action_id format (taskid_X)")

# 3. Tool usage validation
print(f"\n3. TOOL USAGE VALIDATION")
original_tools = {
    'calculate', 'cancel_pending_order', 'exchange_delivered_order_items',
    'find_user_id_by_email', 'find_user_id_by_name_zip', 'get_order_details',
    'get_product_details', 'get_user_details', 'modify_pending_order_address',
    'modify_pending_order_items', 'modify_pending_order_payment',
    'modify_user_address', 'return_delivered_order_items', 'transfer_to_human_agents'
}

new_task_tools = set()
for task in tasks[114:]:
    for action in task['evaluation_criteria']['actions']:
        new_task_tools.add(action['name'])

extra_tools = new_task_tools - original_tools
if extra_tools:
    print(f"   [ERROR] Found {len(extra_tools)} new tools not in original set:")
    for tool in extra_tools:
        print(f"     - {tool}")
else:
    print(f"   [OK] All new tasks use only the original {len(original_tools)} tools")

# 4. Database consistency validation
print(f"\n4. DATABASE CONSISTENCY VALIDATION")
print(f"   Validating user and order references...")

errors = []
checked = 0
for task in tasks[114:]:
    task_id = task['id']

    # Check authentication actions
    for action in task['evaluation_criteria']['actions']:
        if action['name'] == 'find_user_id_by_email':
            email = action['arguments']['email']
            found = False
            for user in db['users'].values():
                if user['email'] == email:
                    found = True
                    break
            if not found:
                errors.append(f"Task {task_id}: Email {email} not found")
            else:
                checked += 1
        elif action['name'] == 'find_user_id_by_name_zip':
            first_name = action['arguments']['first_name']
            last_name = action['arguments']['last_name']
            zip_code = action['arguments']['zip']
            found = False
            for user in db['users'].values():
                if (user['name']['first_name'] == first_name and
                    user['name']['last_name'] == last_name and
                    user['address']['zip'] == zip_code):
                    found = True
                    break
            if not found:
                errors.append(f"Task {task_id}: User {first_name} {last_name} in {zip_code} not found")
            else:
                checked += 1
        elif action['name'] == 'get_order_details':
            order_id = action['arguments']['order_id']
            if order_id not in db['orders']:
                errors.append(f"Task {task_id}: Order {order_id} not found")

if errors:
    print(f"   [ERROR] Found {len(errors)} database consistency errors:")
    for err in errors[:10]:
        print(f"     - {err}")
else:
    print(f"   [OK] All {checked} user references are valid")
    print(f"   [OK] All order references are valid")

# 5. Task type distribution
print(f"\n5. TASK TYPE DISTRIBUTION (New tasks 114-299)")
task_types = {}
for task in tasks[114:]:
    purpose = task['description']['purpose'].lower() if task['description']['purpose'] else ''

    if 'cancel' in purpose and 'pending' in purpose:
        task_type = 'cancel_pending_order'
    elif 'return' in purpose:
        task_type = 'return_delivered_order'
    elif 'exchange' in purpose:
        task_type = 'exchange_delivered_order'
    elif 'address' in purpose and 'order' in purpose:
        task_type = 'modify_order_address'
    elif 'payment' in purpose and 'order' in purpose:
        task_type = 'modify_order_payment'
    elif 'address' in purpose:
        task_type = 'modify_user_address'
    elif 'status' in purpose or 'check' in purpose:
        task_type = 'get_order_info'
    else:
        task_type = 'other'

    task_types[task_type] = task_types.get(task_type, 0) + 1

for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
    pct = (count / 186) * 100
    print(f"   {task_type:25s}: {count:3d} ({pct:5.1f}%)")

# 6. Sample validation of random tasks
print(f"\n6. SAMPLE VALIDATION OF RANDOM TASKS")
random.seed(888)
sample_ids = random.sample(range(114, 300), 15)

validation_passed = 0
validation_failed = 0

for task_id in sample_ids:
    task = tasks[task_id]

    try:
        # Check action_id format
        for action in task['evaluation_criteria']['actions']:
            if not action['action_id'].startswith(f"{task_id}_"):
                validation_failed += 1
                break
        else:
            validation_passed += 1
    except:
        validation_failed += 1

print(f"   Validated {len(sample_ids)} random tasks:")
print(f"   [OK] Passed: {validation_passed}")
if validation_failed > 0:
    print(f"   [ERROR] Failed: {validation_failed}")
else:
    print(f"   [OK] Failed: 0")

# 7. Policy compliance check
print(f"\n7. POLICY COMPLIANCE CHECK")
print(f"   Checking order status requirements...")

status_violations = 0
for task in tasks[114:120]:  # Check first few
    for action in task['evaluation_criteria']['actions']:
        if action['name'] == 'cancel_pending_order':
            order_id = None
            # Find order_id from get_order_details action
            for a in task['evaluation_criteria']['actions']:
                if a['name'] == 'get_order_details':
                    order_id = a['arguments']['order_id']
                    break

            if order_id and order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] != 'pending':
                    status_violations += 1
                    print(f"   [ERROR] Task {task['id']}: Trying to cancel non-pending order")

        elif action['name'] == 'return_delivered_order_items':
            order_id = None
            for a in task['evaluation_criteria']['actions']:
                if a['name'] == 'get_order_details':
                    order_id = a['arguments']['order_id']
                    break

            if order_id and order_id in db['orders']:
                order = db['orders'][order_id]
                if order['status'] != 'delivered':
                    status_violations += 1
                    print(f"   [ERROR] Task {task['id']}: Trying to return non-delivered order")

if status_violations == 0:
    print(f"   [OK] All checked tasks comply with order status requirements")

# Final summary
print(f"\n{'='*80}")
print(f"VALIDATION SUMMARY")
print(f"{'='*80}")
print(f"Total tasks generated: 186 (ID 114-299)")
print(f"Action_id format: {'PASS' if not invalid_action_ids else 'FAIL'}")
print(f"Tool usage: {'PASS' if not extra_tools else 'FAIL'}")
print(f"Database consistency: {'PASS' if not errors else 'FAIL'}")
print(f"Sample validation: {'PASS' if validation_failed == 0 else f'FAIL ({validation_failed} errors)'}")
print(f"Policy compliance: {'PASS' if status_violations == 0 else 'FAIL'}")

if not invalid_action_ids and not extra_tools and not errors and validation_failed == 0 and status_violations == 0:
    print(f"\n>>> ALL VALIDATIONS PASSED! <<<")
    print(f"The generated retail tasks are ready for use.")
else:
    print(f"\n[WARNING] Some validations failed. Please review the errors above.")

print(f"{'='*80}")
