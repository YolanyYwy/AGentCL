import json

# Load data
with open(R'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(R'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

print("Validating generated tasks...")
print(f"Total tasks: {len(tasks)}")

# Validate a few sample tasks
sample_ids = [50, 100, 150, 200, 250, 299]

for task_id in sample_ids:
    if task_id >= len(tasks):
        continue

    task = tasks[task_id]
    print(f"\n--- Validating Task {task_id} ---")
    print(f"Purpose: {task['description']['purpose']}")

    # Extract user_id from known_info
    known_info = task['user_scenario']['instructions']['known_info']
    if 'user id is' in known_info:
        user_id = known_info.split('user id is ')[1].split('.')[0].strip()
        print(f"User ID: {user_id}")
        print(f"User exists in db: {user_id in db['users']}")

        # Check if reservation is mentioned
        if 'reservation number is' in known_info:
            res_id = known_info.split('reservation number is ')[1].split('.')[0].strip()
            print(f"Reservation ID: {res_id}")
            print(f"Reservation exists in db: {res_id in db['reservations']}")

            if res_id in db['reservations']:
                res = db['reservations'][res_id]
                print(f"Reservation belongs to user: {res['user_id'] == user_id}")
                print(f"Reservation cabin: {res['cabin']}")
                print(f"Reservation flights: {len(res['flights'])}")

    # Check action_id format
    actions = task['evaluation_criteria']['actions']
    print(f"Number of actions: {len(actions)}")
    for action in actions:
        action_id = action['action_id']
        expected_prefix = f"{task_id}_"
        if not action_id.startswith(expected_prefix):
            print(f"  ERROR: action_id {action_id} doesn't match expected format {expected_prefix}X")
        else:
            print(f"  ✓ action_id {action_id} format correct")

print("\n\n=== Summary ===")
print(f"Total tasks: {len(tasks)}")
print(f"Task IDs range: {tasks[0]['id']} to {tasks[-1]['id']}")

# Count tools used
tools = set()
for task in tasks:
    for action in task.get('evaluation_criteria', {}).get('actions', []):
        tools.add(action['name'])

print(f"Total unique tools: {len(tools)}")
print("Tools used:")
for tool in sorted(tools):
    print(f"  - {tool}")
