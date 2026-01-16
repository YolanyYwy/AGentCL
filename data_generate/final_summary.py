import json

tasks = json.load(open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8'))

print('='*60)
print('FINAL SUMMARY')
print('='*60)
print(f'Total tasks: {len(tasks)}')
print(f'Task ID range: {tasks[0]["id"]} to {tasks[-1]["id"]}')

print(f'\nLast 5 tasks:')
for i in range(-5, 0):
    task = tasks[i]
    print(f'  Task {task["id"]}: {task["description"]["purpose"]}')

print(f'\nSample of new tasks (50, 100, 150, 200, 250, 299):')
for task_id in [50, 100, 150, 200, 250, 299]:
    task = tasks[task_id]
    print(f'\n  Task {task["id"]}:')
    print(f'    Purpose: {task["description"]["purpose"]}')
    print(f'    Actions: {len(task["evaluation_criteria"]["actions"])}')
    action_names = [a["name"] for a in task["evaluation_criteria"]["actions"]]
    print(f'    Tools: {", ".join(action_names)}')

# Count task types
task_types = {}
for task in tasks[50:]:
    purpose = task["description"]["purpose"]
    if "book" in purpose.lower():
        task_type = "book_flight"
    elif "cancel" in purpose.lower():
        task_type = "cancel_reservation"
    elif "change flight" in purpose.lower() or "modify" in purpose.lower():
        task_type = "modify_flight"
    elif "baggage" in purpose.lower() or "bag" in purpose.lower():
        task_type = "add_baggage"
    elif "cabin" in purpose.lower():
        task_type = "change_cabin"
    elif "compensation" in purpose.lower() or "complain" in purpose.lower():
        task_type = "compensation"
    elif "passenger" in purpose.lower():
        task_type = "update_passenger"
    else:
        task_type = "other"

    task_types[task_type] = task_types.get(task_type, 0) + 1

print(f'\n\nTask type distribution (new tasks 50-299):')
for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True):
    print(f'  {task_type}: {count}')

print('\n' + '='*60)
