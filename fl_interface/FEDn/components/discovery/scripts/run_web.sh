find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
find . -path "*/migrations/*.pyc"  -delete

sleep 5

echo "====== MakeMigration ===="
python3 manage.py makemigrations combiner

echo "====== Migrate ===="
python3 manage.py migrate

echo "====== Load Data ===="
python3 manage.py loaddata seed.json

echo "====== Run Server ==="
python3 manage.py runserver 0.0.0.0:${DISCOVERY_PORT}
