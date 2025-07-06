# This for AIRFLOW__CORE__FERNET_KEY
from cryptography.fernet import Fernet
fernet_key = Fernet.generate_key()
print(fernet_key.decode())
# This for AIRFLOW__WEBSERVER__SECRET_KEY
import os
print(os.urandom(16))