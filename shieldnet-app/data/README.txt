==============================
GUÍA DE INSTALACIÓN SHIELDNET-APP
==============================

🖥️ SISTEMA OPERATIVO: Windows
📦 ENTORNO: Python + Flask + XAMPP (MySQL)

----------------------------------
1️⃣ INSTALAR PYTHON
----------------------------------
1. Descarga Python desde: https://www.python.org/downloads/
2. Durante la instalación, marca la opción:
   ✅ "Add Python to PATH"
3. Verifica la instalación:
   CMD → python --version

----------------------------------
2️⃣ CREAR EL PROYECTO
----------------------------------
1. Crea una carpeta del proyecto:
   mkdir C:\shieldnet-app
   cd C:\shieldnet-app

2. Crea el entorno virtual:
   python -m venv venv 
   
3. Activa el entorno virtual:
   venv\Scripts\activate

(Debería verse así al inicio de la línea)
   (venv) C:\shieldnet-app>

----------------------------------
3️⃣ INSTALAR DEPENDENCIAS
----------------------------------
Ejecutar dentro del entorno virtual:

pip install flask flask-cors mysql-connector-python scikit-learn joblib
pip install --upgrade pip
pip install pandas numpy scikit-learn joblib
pip install flask scikit-learn joblib sqlalchemy pymysql
pip install flask sqlalchemy pymysql scikit-learn joblib pandas numpy
pip install pandas
pip freeze > requirements.txt

📦 Explicación de librerías:
- Flask → framework web backend
- flask-cors → permite peticiones desde el frontend
- mysql-connector-python → conexión con base de datos XAMPP
- scikit-learn → librería para IA / Machine Learning
- joblib → guardar y cargar el modelo entrenado
- pandas → manejo de datos y textos
- pip freeze > requirements.txt → sirve para mover el proyecto, puedes reinstalar todo con (pip install -r requirements.txt
)


----------------------------------
4️⃣ ESTRUCTURA DEL PROYECTO
----------------------------------
C:\shieldnet-app\
│
├── templates\
│   └── index.html        ← Interfaz HTML (frontend)
│
├── venv\                 ← Entorno virtual (no modificar)
│
├── app.py                ← Backend Flask
├── train_model.py        ← Script para entrenar IA
├── model_shieldnet.joblib← Modelo IA entrenado
└── .gitignore (opcional)

----------------------------------
5️⃣ ENTRENAR EL MODELO IA
----------------------------------
(Con el entorno virtual activado)

python train_model.py

----------------------------------
6️⃣ EJECUTAR LA APP
----------------------------------
(Una vez tengas app.py listo)
python app.py

Luego, abre en tu navegador:
http://127.0.0.1:5000/

----------------------------------
7️⃣ OPCIONAL: GUARDAR DEPENDENCIAS
----------------------------------
Para registrar las librerías instaladas:
pip freeze > requirements.txt

Luego, en otro equipo puedes instalar todo con:
pip install -r requirements.txt
