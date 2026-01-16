import json

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

# Additional random tasks
task_ids = [63, 118, 72, 246, 154]

print("="*80)
print("VALIDATING ADDITIONAL RANDOM TASKS")
print("="*80)

for task_id in task_ids:
    task = tasks[task_id]
    print(f"\n{'='*80}")
    print(f"TASK {task_id}: {task['description']['purpose']}")
    print(f"{'='*80}")

    known_info = task['user_scenario']['instructions']['known_info']
    user_id = known_info.split('user id is ')[1].split('.')[0].split('\n')[0].strip()

    print(f"\n[CHECK] User: {user_id}")
    if user_id in db['users']:
        user = db['users'][user_id]
        print(f"  [OK] User exists")
        print(f"  Name: {user['name']['first_name']} {user['name']['last_name']}")
        print(f"  Membership: {user['membership']}")

        # Check if task involves a reservation
        if 'reservation number is' in known_info:
            res_id = known_info.split('reservation number is ')[1].split('.')[0].split('\n')[0].strip()
            print(f"\n[CHECK] Reservation: {res_id}")

            if res_id in db['reservations']:
                res = db['reservations'][res_id]
                print(f"  [OK] Reservation exists")
                print(f"  User match: {res['user_id'] == user_id}")
                print(f"  Route: {res['origin']} -> {res['destination']}")
                print(f"  Cabin: {res['cabin']}")
                print(f"  Created: {res['created_at']}")
                print(f"  Insurance: {res.get('insurance', 'no')}")

                # For cancellation tasks, check eligibility
                if 'cancel' in task['description']['purpose'].lower():
                    from datetime import datetime
                    created_at = datetime.fromisoformat(res['created_at'].replace('T', ' '))
                    current_time = datetime(2024, 5, 15, 15, 0, 0)
                    hours_since = (current_time - created_at).total_seconds() / 3600

                    within_24h = hours_since < 24
                    is_business = res['cabin'] == 'business'
                    has_insurance = res.get('insurance', 'no') == 'yes'
                    can_cancel = within_24h or is_business or has_insurance

                    print(f"\n  [CANCELLATION CHECK]")
                    print(f"    Hours since booking: {hours_since:.1f}")
                    print(f"    Within 24h: {within_24h}")
                    print(f"    Business class: {is_business}")
                    print(f"    Has insurance: {has_insurance}")
                    print(f"    Can cancel: {can_cancel}")

                    # Check if task expects cancellation to be allowed/denied
                    if 'should be allowed' in task['description']['purpose'].lower():
                        if can_cancel:
                            print(f"    [OK] Task expects allowed, and it is allowed")
                        else:
                            print(f"    [ERROR] Task expects allowed, but it should be denied!")
                    elif 'should be denied' in task['description']['purpose'].lower():
                        if not can_cancel:
                            print(f"    [OK] Task expects denied, and it is denied")
                        else:
                            print(f"    [ERROR] Task expects denied, but it should be allowed!")

                # For cabin change tasks
                if 'cabin' in task['description']['purpose'].lower():
                    print(f"\n  [CABIN CHANGE CHECK]")
                    print(f"    Current cabin: {res['cabin']}")
                    # Check if flights are in future
                    all_future = all(f['date'] >= '2024-05-15' for f in res['flights'])
                    print(f"    All flights in future: {all_future}")
                    print(f"    Can change cabin: {all_future}")

                # Verify flights
                print(f"\n  [FLIGHT VERIFICATION]")
                for i, flight_info in enumerate(res['flights']):
                    flight_num = flight_info['flight_number']
                    date = flight_info['date']
                    found = False
                    for f in db['flights'].values():
                        if f['flight_number'] == flight_num:
                            if date in f['dates']:
                                status = f['dates'][date]['status']
                                print(f"    Flight {i+1}: {flight_num} on {date} - {status} [OK]")
                                found = True
                            break
                    if not found:
                        print(f"    Flight {i+1}: {flight_num} on {date} - NOT FOUND [ERROR]")
            else:
                print(f"  [ERROR] Reservation {res_id} not found!")
        else:
            print(f"\n  (No reservation - new booking task)")

        # Validate actions
        print(f"\n[CHECK] Actions:")
        for action in task['evaluation_criteria']['actions']:
            action_id = action['action_id']
            expected_prefix = f"{task_id}_"
            if action_id.startswith(expected_prefix):
                print(f"  [OK] {action_id}: {action['name']}")
            else:
                print(f"  [ERROR] {action_id}: Invalid format (expected {expected_prefix}X)")

        print(f"\n>>> TASK {task_id} VALIDATION COMPLETE <<<")
    else:
        print(f"  [ERROR] User not found!")

print(f"\n{'='*80}")
print("ALL VALIDATIONS COMPLETE")
print(f"{'='*80}")
