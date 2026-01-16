import json
import random

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

# Select random tasks from the new ones (50-299)
random.seed(42)
random_task_ids = random.sample(range(50, 300), 10)
random_task_ids.sort()

print("="*80)
print("VALIDATING RANDOM TASKS AGAINST DB.JSON")
print("="*80)

validation_results = {
    'passed': 0,
    'failed': 0,
    'issues': []
}

for task_id in random_task_ids:
    task = tasks[task_id]
    print(f"\n{'='*80}")
    print(f"TASK {task_id}: {task['description']['purpose']}")
    print(f"{'='*80}")

    known_info = task['user_scenario']['instructions']['known_info']

    # Extract user_id
    if 'user id is' in known_info:
        user_id = known_info.split('user id is ')[1].split('.')[0].split('\n')[0].strip()
        print(f"\n[OK] User ID: {user_id}")

        # Validate user exists
        if user_id in db['users']:
            user = db['users'][user_id]
            print(f"  [OK] User exists in database")
            print(f"  - Name: {user['name']['first_name']} {user['name']['last_name']}")
            print(f"  - Membership: {user['membership']}")
            print(f"  - Payment methods: {len(user['payment_methods'])}")
        else:
            print(f"  [ERROR] User {user_id} NOT FOUND in database!")
            validation_results['failed'] += 1
            validation_results['issues'].append(f"Task {task_id}: User {user_id} not found")
            continue

    # Extract reservation_id if present
    if 'reservation number is' in known_info or 'reservation' in known_info.lower():
        try:
            res_id = known_info.split('reservation number is ')[1].split('.')[0].split('\n')[0].strip()
            print(f"\n[OK] Reservation ID: {res_id}")

            # Validate reservation exists
            if res_id in db['reservations']:
                reservation = db['reservations'][res_id]
                print(f"  [OK] Reservation exists in database")
                print(f"  - Belongs to user: {reservation['user_id']}")

                # Validate user-reservation match
                if reservation['user_id'] == user_id:
                    print(f"  [OK] Reservation belongs to the correct user")
                else:
                    print(f"  [ERROR] Reservation belongs to {reservation['user_id']}, not {user_id}!")
                    validation_results['failed'] += 1
                    validation_results['issues'].append(f"Task {task_id}: User-reservation mismatch")
                    continue

                print(f"  - Origin: {reservation['origin']}")
                print(f"  - Destination: {reservation['destination']}")
                print(f"  - Cabin: {reservation['cabin']}")
                print(f"  - Flight type: {reservation['flight_type']}")
                print(f"  - Number of flights: {len(reservation['flights'])}")
                print(f"  - Number of passengers: {len(reservation['passengers'])}")
                print(f"  - Total baggages: {reservation['total_baggages']}")
                print(f"  - Insurance: {reservation.get('insurance', 'no')}")
                print(f"  - Created at: {reservation['created_at']}")

                # Validate flights in reservation
                print(f"\n  Flight details:")
                for i, flight_info in enumerate(reservation['flights']):
                    flight_num = flight_info['flight_number']
                    date = flight_info['date']
                    print(f"    Flight {i+1}: {flight_num} on {date}")

                    # Find flight in db
                    flight_found = False
                    for f in db['flights'].values():
                        if f['flight_number'] == flight_num:
                            flight_found = True
                            if date in f['dates']:
                                status = f['dates'][date]['status']
                                print(f"      [OK] Flight exists, status: {status}")
                                print(f"      - Route: {f['origin']} -> {f['destination']}")
                            else:
                                print(f"      [ERROR] Date {date} not found for flight {flight_num}")
                            break

                    if not flight_found:
                        print(f"      [ERROR] Flight {flight_num} not found in database!")

                # Validate task-specific logic
                purpose = task['description']['purpose'].lower()

                if 'cancel' in purpose:
                    print(f"\n  Task type: CANCEL RESERVATION")
                    # Check cancellation eligibility
                    from datetime import datetime
                    created_at = datetime.fromisoformat(reservation['created_at'].replace('T', ' '))
                    current_time = datetime(2024, 5, 15, 15, 0, 0)
                    hours_since_booking = (current_time - created_at).total_seconds() / 3600

                    within_24h = hours_since_booking < 24
                    is_business = reservation['cabin'] == 'business'
                    has_insurance = reservation.get('insurance', 'no') == 'yes'

                    print(f"    - Hours since booking: {hours_since_booking:.1f}")
                    print(f"    - Within 24h: {within_24h}")
                    print(f"    - Business class: {is_business}")
                    print(f"    - Has insurance: {has_insurance}")

                    can_cancel = within_24h or is_business or has_insurance
                    print(f"    - Can cancel: {can_cancel}")

                elif 'modify' in purpose or 'change flight' in purpose:
                    print(f"\n  Task type: MODIFY FLIGHT")
                    is_basic_economy = reservation['cabin'] == 'basic_economy'
                    print(f"    - Basic economy: {is_basic_economy}")
                    print(f"    - Can modify: {not is_basic_economy}")

                elif 'baggage' in purpose or 'bag' in purpose:
                    print(f"\n  Task type: ADD BAGGAGE")
                    membership = user.get('membership', 'regular')
                    cabin = reservation['cabin']

                    # Calculate free baggage allowance
                    free_bags_map = {
                        'regular': {'basic_economy': 0, 'economy': 1, 'business': 2},
                        'silver': {'basic_economy': 1, 'economy': 2, 'business': 3},
                        'gold': {'basic_economy': 2, 'economy': 3, 'business': 4}
                    }

                    num_passengers = len(reservation['passengers'])
                    free_bags_per_passenger = free_bags_map[membership][cabin]
                    total_free_bags = free_bags_per_passenger * num_passengers

                    print(f"    - Membership: {membership}")
                    print(f"    - Cabin: {cabin}")
                    print(f"    - Passengers: {num_passengers}")
                    print(f"    - Free bags per passenger: {free_bags_per_passenger}")
                    print(f"    - Total free bags: {total_free_bags}")
                    print(f"    - Current total bags: {reservation['total_baggages']}")

                elif 'cabin' in purpose:
                    print(f"\n  Task type: CHANGE CABIN")
                    current_cabin = reservation['cabin']
                    print(f"    - Current cabin: {current_cabin}")

                    # Check if any flight has been flown
                    all_future = True
                    for flight_info in reservation['flights']:
                        date = flight_info['date']
                        if date < '2024-05-15':
                            all_future = False
                            break

                    print(f"    - All flights in future: {all_future}")
                    print(f"    - Can change cabin: {all_future}")

                elif 'compensation' in purpose or 'complain' in purpose:
                    print(f"\n  Task type: COMPENSATION")
                    membership = user.get('membership', 'regular')
                    has_insurance = reservation.get('insurance', 'no') == 'yes'
                    is_business = reservation['cabin'] == 'business'

                    eligible = (membership in ['silver', 'gold']) or has_insurance or is_business

                    print(f"    - Membership: {membership}")
                    print(f"    - Has insurance: {has_insurance}")
                    print(f"    - Business class: {is_business}")
                    print(f"    - Eligible for compensation: {eligible}")

            else:
                print(f"  [ERROR] Reservation {res_id} NOT FOUND in database!")
                validation_results['failed'] += 1
                validation_results['issues'].append(f"Task {task_id}: Reservation {res_id} not found")
                continue
        except:
            # No reservation in this task (e.g., booking new flight)
            print(f"\n  (No existing reservation - this is a new booking task)")

    # Validate action_id format
    print(f"\n[OK] Actions:")
    actions = task['evaluation_criteria']['actions']
    for action in actions:
        action_id = action['action_id']
        expected_prefix = f"{task_id}_"
        if action_id.startswith(expected_prefix):
            print(f"  [OK] {action_id}: {action['name']}")
        else:
            print(f"  [ERROR] action_id {action_id} doesn't match expected format {expected_prefix}X")
            validation_results['failed'] += 1
            validation_results['issues'].append(f"Task {task_id}: Invalid action_id format {action_id}")

    # If we got here without errors, mark as passed
    if not any(f"Task {task_id}:" in issue for issue in validation_results['issues']):
        validation_results['passed'] += 1
        print(f"\n>>> TASK {task_id} VALIDATION PASSED <<<")

# Final summary
print(f"\n\n{'='*80}")
print("VALIDATION SUMMARY")
print(f"{'='*80}")
print(f"Tasks validated: {len(random_task_ids)}")
print(f"Passed: {validation_results['passed']}")
print(f"Failed: {validation_results['failed']}")

if validation_results['issues']:
    print(f"\nIssues found:")
    for issue in validation_results['issues']:
        print(f"  - {issue}")
else:
    print(f"\n>>> ALL VALIDATIONS PASSED! <<<")

print(f"\nRandom task IDs validated: {random_task_ids}")
