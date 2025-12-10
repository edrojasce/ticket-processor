#!/usr/bin/env python3
"""
Script de diagnóstico para Ollama
Verifica estado, modelos y capacidades del sistema
"""

import requests
import json
import base64
from PIL import Image
import io

def check_ollama_connection():
    """Verifica conexión básica"""
    print("\n" + "="*60)
    print("1. VERIFICANDO CONEXIÓN A OLLAMA")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("[+] Ollama está corriendo")
            print(f"    Version: {response.json().get('version', 'unknown')}")
            return True
        else:
            print(f"[!] Ollama responde pero con error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[!] ERROR: No se puede conectar a Ollama")
        print("    Asegúrate de ejecutar: ollama serve")
        return False
    except Exception as e:
        print(f"[!] Error inesperado: {e}")
        return False


def list_models():
    """Lista modelos disponibles"""
    print("\n" + "="*60)
    print("2. MODELOS DISPONIBLES")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if not models:
                print("[!] No hay modelos instalados")
                print("\n    Modelos recomendados para tickets:")
                print("    - ollama pull llava:7b (más rápido, 8GB RAM)")
                print("    - ollama pull llava:13b (mejor calidad, 16GB RAM)")
                print("    - ollama pull llama3.2-vision:11b (máxima calidad, 16GB RAM)")
                return []
            
            print(f"[+] Encontrados {len(models)} modelos:\n")
            for model in models:
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)  # GB
                modified = model.get('modified_at', '')
                print(f"    - {name}")
                print(f"      Tamaño: {size:.2f} GB")
                print(f"      Modificado: {modified[:10]}")
                print()
            
            return [m['name'] for m in models]
        else:
            print(f"[!] Error al listar modelos: {response.status_code}")
            return []
    except Exception as e:
        print(f"[!] Error: {e}")
        return []


def check_running_models():
    """Verifica qué modelos están cargados en memoria"""
    print("\n" + "="*60)
    print("3. MODELOS EN MEMORIA")
    print("="*60)
    
    try:
        response = requests.get("http://localhost:11434/api/ps", timeout=5)
        if response.status_code == 200:
            running = response.json().get('models', [])
            if not running:
                print("[*] No hay modelos cargados en memoria")
                print("    (El primer uso será lento mientras carga)")
                return []
            
            print(f"[+] Modelos cargados: {len(running)}\n")
            for model in running:
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)
                print(f"    - {name} ({size:.2f} GB en RAM)")
            return running
        else:
            print(f"[!] No se pudo verificar: {response.status_code}")
            return []
    except Exception as e:
        print(f"[!] Error: {e}")
        return []


def test_vision_model(model_name):
    """Prueba un modelo de visión con imagen simple"""
    print("\n" + "="*60)
    print(f"4. PROBANDO MODELO: {model_name}")
    print("="*60)
    
    # Crear imagen de prueba simple
    img = Image.new('RGB', (200, 100), color='white')
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), "TOTAL: $123.45", fill='black')
    
    # Convertir a base64
    buffered = io.BytesIO()
    img.save(buffered, format='JPEG')
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    print("[*] Enviando imagen de prueba simple...")
    print("    (Imagen de prueba dice: 'TOTAL: $123.45')")
    
    try:
        payload = {
            "model": model_name,
            "prompt": "What text do you see in this image? Just read the text.",
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 100,
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '')
            print(f"[+] Modelo respondió correctamente")
            print(f"    Respuesta: {text[:200]}")
            return True
        else:
            print(f"[!] Error {response.status_code}")
            try:
                error_msg = response.json().get('error', 'unknown')
                print(f"    Mensaje: {error_msg}")
            except:
                print(f"    Raw: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print("[!] Timeout - El modelo tardó más de 60 segundos")
        print("    Esto puede indicar falta de RAM/VRAM")
        return False
    except Exception as e:
        print(f"[!] Error inesperado: {e}")
        return False


def recommend_model(available_models):
    """Recomienda mejor modelo basado en disponibles"""
    print("\n" + "="*60)
    print("5. RECOMENDACIÓN")
    print("="*60)
    
    vision_models = [m for m in available_models if 'vision' in m.lower() or 'llava' in m.lower() or 'bakllava' in m.lower()]
    
    if not vision_models:
        print("[!] No tienes modelos de visión instalados")
        print("\n[+] Recomendación según tu RAM:\n")
        print("    8GB RAM o menos:")
        print("    → ollama pull llava:7b")
        print("\n    16GB RAM:")
        print("    → ollama pull llava:13b")
        print("\n    32GB+ RAM:")
        print("    → ollama pull llama3.2-vision:11b")
        return None
    
    print(f"[+] Tienes {len(vision_models)} modelos de visión:\n")
    
    # Ordenar por preferencia
    preference = {
        'llama3.2-vision:11b': 1,
        'llava:13b': 2,
        'llava:7b': 3,
        'bakllava': 4,
    }
    
    def get_priority(name):
        for key, priority in preference.items():
            if key in name:
                return priority
        return 999
    
    vision_models.sort(key=get_priority)
    
    for i, model in enumerate(vision_models, 1):
        print(f"    {i}. {model}")
        if i == 1:
            print(f"       ← RECOMENDADO (mejor disponible)")
    
    return vision_models[0] if vision_models else None


def main():
    print("\n" + "="*60)
    print("DIAGNÓSTICO DE OLLAMA PARA EXPENSE TRACKER")
    print("="*60)
    
    # 1. Verificar conexión
    if not check_ollama_connection():
        print("\n[!] Ollama no está corriendo. Inicia con: ollama serve")
        return
    
    # 2. Listar modelos
    models = list_models()
    
    # 3. Ver modelos en memoria
    check_running_models()
    
    # 4. Recomendar modelo
    recommended = recommend_model(models)
    
    # 5. Probar modelo recomendado
    if recommended:
        print(f"\n[*] Probando modelo recomendado: {recommended}")
        input("    Presiona Enter para continuar (esto puede tardar)...")
        if test_vision_model(recommended):
            print(f"\n[+] ¡Éxito! Usa este modelo en main_AI.py:")
            print(f"    model = '{recommended}'")
        else:
            print(f"\n[!] El modelo falló. Intenta con otro modelo más pequeño.")
    else:
        print("\n[!] Necesitas instalar un modelo de visión primero.")
    
    print("\n" + "="*60)
    print("DIAGNÓSTICO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()