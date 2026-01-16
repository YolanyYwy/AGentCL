import json

# Load data
with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\db.json', 'r', encoding='utf-8') as f:
    db = json.load(f)

with open(r'D:\Desktop\work\tau2-bench\data\tau2\domains\airline\tasks.json', 'r', encoding='utf-8') as f:
    tasks = json.load(f)

print('Validating task 50:')
task = tasks[50]
user_id = task['user_scenario']['instructions']['known_info'].split('user id is ')[1].split('.')[0]
res_id = task['user_scenario']['instructions']['known_info'].split('reservation number is ')[1].split('.')[0]
print(f'User ID: {user_id}')
print(f'Reservation ID: {res_id}')
print(f'User exists: {user_id in db["users"]}')
print(f'Reservation exists: {res_id in db["reservations"]}')

if res_id in db['reservations']:
    res = db['reservations'][res_id]
    print(f'Reservation user_id: {res["user_id"]}')
    print(f'Match: {res["user_id"] == user_id}')
    print(f'Reservation details:')
    print(f'  Origin: {res["origin"]}')
    print(f'  Destination: {res["destination"]}')
    print(f'  Cabin: {res["cabin"]}')
    print(f'  Flights: {res["flights"]}')
    print(f'  Total baggages: {res["total_baggages"]}')

print('\n\nValidating task 100:')
task = tasks[100]
known_info = task['user_scenario']['instructions']['known_info']
user_id = known_info.split('user id is ')[1].split('.')[0]
res_id = known_info.split('reservation number is ')[1].split('.')[0]
print(f'User ID: {user_id}')
print(f'Reservation ID: {res_id}')
print(f'User exists: {user_id in db["users"]}')
print(f'Reservation exists: {res_id in db["reservations"]}')

if res_id in db['reservations']:
    res = db['reservations'][res_id]
    print(f'Reservation user_id: {res["user_id"]}')
    print(f'Match: {res["user_id"] == user_id}')
    print(f'Cabin: {res["cabin"]}')
    print(f'Can modify (not basic_economy): {res["cabin"] != "basic_economy"}')
