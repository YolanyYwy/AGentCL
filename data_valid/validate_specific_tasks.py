import json

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

# Specific tasks to validate
task_ids = [155, 189, 225, 267, 290]

print("="*80)
print("DETAILED VALIDATION OF SPECIFIC TASKS")
print("="*80)

for task_id in task_ids:
    task = tasks[task_id]
    print(f"\n{'='*80}")
    print(f"TASK {task_id}: {task['description']['purpose']}")
    print(f"{'='*80}")

    known_info = task['user_scenario']['instructions']['known_info']
    user_id = known_info.split('user id is ')[1].split('.')[0].split('\n')[0].strip()

    print(f"\nUser: {user_id}")
    if user_id in db['users']:
        user = db['users'][user_id]
        print(f"  Name: {user['name']['first_name']} {user['name']['last_name']}")
        print(f"  Membership: {user['membership']}")
        print(f"  Reservations: {user.get('reservations', [])}")

        # Check if task involves a reservation
        if 'reservation number is' in known_info:
            res_id = known_info.split('reservation number is ')[1].split('.')[0].split('\n')[0].strip()
            print(f"\nReservation: {res_id}")

            if res_id in db['reservations']:
                res = db['reservations'][res_id]
                print(f"  User match: {res['user_id'] == user_id} (res user: {res['user_id']})")
                print(f"  Route: {res['origin']} -> {res['destination']}")
                print(f"  Cabin: {res['cabin']}")
                print(f"  Passengers: {len(res['passengers'])}")
                print(f"  Flights:")
                for flight in res['flights']:
                    print(f"    - {flight['flight_number']} on {flight['date']}")

                # Verify flights exist
                for flight_info in res['flights']:
                    flight_num = flight_info['flight_number']
                    date = flight_info['date']
                    found = False
                    for f in db['flights'].values():
                        if f['flight_number'] == flight_num:
                            if date in f['dates']:
                                print(f"      [OK] {flight_num} on {date}: {f['dates'][date]['status']}")
                                found = True
                            break
                    if not found:
                        print(f"      [ERROR] {flight_num} on {date}: NOT FOUND")
            else:
                print(f"  [ERROR] Reservation {res_id} not found!")

        # Validate actions
        print(f"\nActions:")
        for action in task['evaluation_criteria']['actions']:
            print(f"  {action['action_id']}: {action['name']}")
            if action['name'] == 'get_user_details':
                arg_user = action['arguments']['user_id']
                print(f"    -> user_id: {arg_user} (match: {arg_user == user_id})")
            elif action['name'] == 'get_reservation_details':
                arg_res = action['arguments']['reservation_id']
                print(f"    -> reservation_id: {arg_res}")
            elif action['name'] == 'search_direct_flight':
                print(f"    -> origin: {action['arguments']['origin']}, destination: {action['arguments']['destination']}")

        print(f"\n[OK] Task {task_id} validated successfully")
    else:
        print(f"  [ERROR] User not found!")

print(f"\n{'='*80}")
print("VALIDATION COMPLETE")
print(f"{'='*80}")
