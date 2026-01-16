import json
import random

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

print("="*80)
print("COMPREHENSIVE VALIDATION REPORT")
print("="*80)

# 1. Basic statistics
print(f"\n1. BASIC STATISTICS")
print(f"   Total tasks: {len(tasks)}")
print(f"   Original tasks (0-49): 50")
print(f"   New tasks (50-299): 250")
print(f"   Task ID range: {tasks[0]['id']} to {tasks[-1]['id']}")

# 2. Validate all action_id formats
print(f"\n2. ACTION_ID FORMAT VALIDATION")
invalid_action_ids = []
for task in tasks[50:]:
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
    print(f"   [OK] All 250 new tasks have correct action_id format (taskid_X)")

# 3. Tool usage validation
print(f"\n3. TOOL USAGE VALIDATION")
original_tools = {
    'book_reservation', 'calculate', 'cancel_reservation',
    'get_reservation_details', 'get_user_details', 'search_direct_flight',
    'send_certificate', 'transfer_to_human_agents',
    'update_reservation_baggages', 'update_reservation_flights',
    'update_reservation_passengers'
}

new_task_tools = set()
for task in tasks[50:]:
    for action in task['evaluation_criteria']['actions']:
        new_task_tools.add(action['name'])

extra_tools = new_task_tools - original_tools
if extra_tools:
    print(f"   [ERROR] Found {len(extra_tools)} new tools not in original set:")
    for tool in extra_tools:
        print(f"     - {tool}")
else:
    print(f"   [OK] All new tasks use only the original {len(original_tools)} tools")
    print(f"   Tools used: {', '.join(sorted(new_task_tools))}")

# 4. Database consistency validation
print(f"\n4. DATABASE CONSISTENCY VALIDATION")
print(f"   Validating user and reservation references...")

errors = []
checked = 0
for task in tasks[50:]:
    task_id = task['id']
    known_info = task['user_scenario']['instructions']['known_info']

    # Check user_id
    if 'user id is' in known_info:
        user_id = known_info.split('user id is ')[1].split('.')[0].split('\n')[0].strip()
        if user_id not in db['users']:
            errors.append(f"Task {task_id}: User {user_id} not found")
        else:
            checked += 1

            # Check reservation_id if present
            if 'reservation number is' in known_info:
                res_id = known_info.split('reservation number is ')[1].split('.')[0].split('\n')[0].strip()
                if res_id not in db['reservations']:
                    errors.append(f"Task {task_id}: Reservation {res_id} not found")
                else:
                    res = db['reservations'][res_id]
                    if res['user_id'] != user_id:
                        errors.append(f"Task {task_id}: User-reservation mismatch")

if errors:
    print(f"   [ERROR] Found {len(errors)} database consistency errors:")
    for err in errors[:10]:
        print(f"     - {err}")
else:
    print(f"   [OK] All {checked} user references are valid")
    print(f"   [OK] All user-reservation pairs match correctly")

# 5. Task type distribution
print(f"\n5. TASK TYPE DISTRIBUTION (New tasks 50-299)")
task_types = {}
for task in tasks[50:]:
    purpose = task['description']['purpose'].lower()
    if 'book' in purpose:
        task_type = 'book_flight'
    elif 'cancel' in purpose:
        task_type = 'cancel_reservation'
    elif 'change flight' in purpose or 'modify' in purpose:
        task_type = 'modify_flight'
    elif 'baggage' in purpose or 'bag' in purpose:
        task_type = 'add_baggage'
    elif 'cabin' in purpose:
        task_type = 'change_cabin'
    elif 'compensation' in purpose or 'complain' in purpose:
        task_type = 'compensation'
    elif 'passenger' in purpose:
        task_type = 'update_passenger'
    else:
        task_type = 'other'

    task_types[task_type] = task_types.get(task_type, 0) + 1

for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
    pct = (count / 250) * 100
    print(f"   {task_type:20s}: {count:3d} ({pct:5.1f}%)")

# 6. Sample validation of random tasks
print(f"\n6. SAMPLE VALIDATION OF RANDOM TASKS")
random.seed(999)
sample_ids = random.sample(range(50, 300), 20)

validation_passed = 0
validation_failed = 0

for task_id in sample_ids:
    task = tasks[task_id]
    known_info = task['user_scenario']['instructions']['known_info']

    try:
        user_id = known_info.split('user id is ')[1].split('.')[0].split('\n')[0].strip()
        if user_id not in db['users']:
            validation_failed += 1
            continue

        # Check reservation if present
        if 'reservation number is' in known_info:
            res_id = known_info.split('reservation number is ')[1].split('.')[0].split('\n')[0].strip()
            if res_id not in db['reservations']:
                validation_failed += 1
                continue

            res = db['reservations'][res_id]
            if res['user_id'] != user_id:
                validation_failed += 1
                continue

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
print(f"   Checking cancellation tasks...")

cancel_tasks = [t for t in tasks[50:] if 'cancel' in t['description']['purpose'].lower()]
print(f"   Total cancellation tasks: {len(cancel_tasks)}")

from datetime import datetime
current_time = datetime(2024, 5, 15, 15, 0, 0)

policy_violations = 0
for task in cancel_tasks[:10]:  # Check first 10
    known_info = task['user_scenario']['instructions']['known_info']
    if 'reservation number is' in known_info:
        res_id = known_info.split('reservation number is ')[1].split('.')[0].split('\n')[0].strip()
        if res_id in db['reservations']:
            res = db['reservations'][res_id]
            created_at = datetime.fromisoformat(res['created_at'].replace('T', ' '))
            hours_since = (current_time - created_at).total_seconds() / 3600

            within_24h = hours_since < 24
            is_business = res['cabin'] == 'business'
            has_insurance = res.get('insurance', 'no') == 'yes'
            can_cancel = within_24h or is_business or has_insurance

            # Check if task expectation matches policy
            expects_allowed = 'should be allowed' in task['description']['purpose'].lower()
            expects_denied = 'should be denied' in task['description']['purpose'].lower()

            if expects_allowed and not can_cancel:
                policy_violations += 1
            elif expects_denied and can_cancel:
                policy_violations += 1

if policy_violations > 0:
    print(f"   [ERROR] Found {policy_violations} policy violations")
else:
    print(f"   [OK] All checked cancellation tasks comply with policy")

# Final summary
print(f"\n{'='*80}")
print(f"VALIDATION SUMMARY")
print(f"{'='*80}")
print(f"Total tasks generated: 250 (ID 50-299)")
print(f"Action_id format: {'PASS' if not invalid_action_ids else 'FAIL'}")
print(f"Tool usage: {'PASS' if not extra_tools else 'FAIL'}")
print(f"Database consistency: {'PASS' if not errors else 'FAIL'}")
print(f"Sample validation: {'PASS' if validation_failed == 0 else f'FAIL ({validation_failed} errors)'}")
print(f"Policy compliance: {'PASS' if policy_violations == 0 else 'FAIL'}")

if not invalid_action_ids and not extra_tools and not errors and validation_failed == 0 and policy_violations == 0:
    print(f"\n>>> ALL VALIDATIONS PASSED! <<<")
    print(f"The generated tasks are ready for use.")
else:
    print(f"\n[WARNING] Some validations failed. Please review the errors above.")

print(f"{'='*80}")
