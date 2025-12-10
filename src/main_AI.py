import json
import base64
import requests
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image
import io

class AIExpenseTracker:
    def __init__(self, model='llava:7b', ollama_url='http://localhost:11434'):
        """
        Inicializa el tracker con IA
        
        Args:
            model: Modelo de Ollama a usar
                   - llama3.2-vision:11b (mejor calidad, más lento, 16GB RAM)
                   - 'llama3:latest' 
                   - 'llama3.2-vision:90b' (máxima calidad, muy lento, 64GB RAM)
                   - 'llava:7b' (más rápido, menos RAM, 8GB RAM)
                   - 'llava:13b' (balance, 16GB RAM)
            ollama_url: URL donde corre Ollama
        """
        self.model = model
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        # Verificar que Ollama esté corriendo
        self._check_ollama_status()
        
        self.categories = {
            'Alimentos': ['oxxo', 'seven', '7-eleven', 'walmart', 'soriana', 'super', 
                         'restaurant', 'cafe', 'food', 'comida', 'papos'],
            'Transporte': ['uber', 'didi', 'gas', 'gasolina', 'taxi', 'parking'],
            'Entretenimiento': ['cinema', 'cine', 'netflix', 'spotify', 'hobby'],
            'Salud': ['farmacia', 'pharmacy', 'hospital', 'doctor'],
            'Servicios': ['cfe', 'telmex', 'telcel', 'internet', 'netpay'],
            'Ropa': ['zara', 'liverpool', 'nike', 'adidas'],
            'Tecnologia': ['apple', 'samsung', 'office depot'],
            'Otros': []
        }
    
    def _check_ollama_status(self):
        """Verifica que Ollama esté corriendo y el modelo disponible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if not any(self.model in name for name in model_names):
                    print(f"\n[!] ADVERTENCIA: Modelo '{self.model}' no encontrado")
                    print(f"[!] Modelos disponibles: {', '.join(model_names)}")
                    print(f"\n[!] Para instalar el modelo ejecuta:")
                    print(f"    ollama pull {self.model}")
                    raise Exception(f"Modelo {self.model} no disponible")
                else:
                    print(f"[+] Ollama conectado - Modelo: {self.model}")
            else:
                raise Exception("Ollama no responde correctamente")
        except requests.exceptions.ConnectionError:
            print("\n[!] ERROR: No se puede conectar a Ollama")
            print("[!] Asegurate de que Ollama este corriendo:")
            print("    - macOS/Linux: ollama serve")
            print("    - Windows: La app de Ollama debe estar abierta")
            raise Exception("Ollama no esta corriendo")
    
    def image_to_base64(self, image_path):
        """Convierte imagen a base64 para enviar a Ollama"""
        try:
            # Abrir y redimensionar si es muy grande
            img = Image.open(image_path)
            
            # Redimensionar para acelerar procesamiento (max 2000px)
            max_size = 2000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convertir a base64
            buffered = io.BytesIO()
            img.save(buffered, format=img.format or 'JPEG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return img_base64
        except Exception as e:
            print(f"[!] Error al procesar imagen {image_path}: {e}")
            return None
    
    def analyze_receipt_with_ai(self, image_path):
        """Analiza un ticket usando IA de visión"""
        print(f"Procesando: {image_path.name}")
        
        # Convertir imagen a base64
        img_base64 = self.image_to_base64(image_path)
        if not img_base64:
            return self._create_error_result(image_path.name, "Error al leer imagen")
        
        # Prompt para el modelo
        prompt = """Analiza este ticket/recibo de compra y extrae la siguiente información.
Responde ÚNICAMENTE con un objeto JSON válido, sin texto adicional antes o después. Este es un ejemplo de la estructura esperada:

{
  "comercio": "nombre del establecimiento",
  "fecha": "YYYY-MM-DD (formato estricto, si encuentras DD/MM/YYYY conviértelo)",
  "hora": "HH:MM",
  "monto": 123.45,
  "metodo_pago": "últimos 4 dígitos de tarjeta (ej: ****1234) o 'Efectivo' o tipo de tarjeta",
  "categoria": "una de estas: Alimentos, Transporte, Entretenimiento, Salud, Servicios, Ropa, Tecnologia, Otros"
}

Reglas importantes:
- Si no encuentras un dato, usa null
- El monto debe ser un número decimal (ej: 123.45, no "$123.45")
- La fecha debe estar en formato YYYY-MM-DD
- Para método de pago: si ves asteriscos seguidos de 4 dígitos, usa ****XXXX
- La categoría debe ser exactamente una de las listadas
- Busca el TOTAL final, no subtotales
- NO incluyas explicaciones, SOLO el JSON"""

        try:
            # Llamar a Ollama API
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "format": "json"  # Forzar respuesta en JSON
            }
            
            print(f"  [*] Analizando con IA (esto puede tomar 10-30 segundos)...")
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=120  # 2 minutos timeout
            )
            
            if response.status_code != 200:
                print(f"  [!] Error en API: {response.status_code}")
                return self._create_error_result(image_path.name, f"API Error {response.status_code}")
            
            # Parsear respuesta
            result = response.json()
            ai_response = result.get('response', '{}')
            
            # Limpiar respuesta (a veces el modelo agrega texto extra)
            ai_response = ai_response.strip()
            if '```json' in ai_response:
                ai_response = ai_response.split('```json')[1].split('```')[0].strip()
            elif '```' in ai_response:
                ai_response = ai_response.split('```')[1].split('```')[0].strip()
            
            # Parse JSON
            try:
                data = json.loads(ai_response)
            except json.JSONDecodeError:
                print(f"  [!] Respuesta no es JSON válido")
                print(f"  [!] Respuesta recibida: {ai_response[:200]}")
                return self._create_error_result(image_path.name, "Respuesta inválida de IA")
            
            # Validar y limpiar datos
            expense = {
                'archivo': image_path.name,
                'fecha': self._validate_date(data.get('fecha')),
                'hora': data.get('hora'),
                'comercio': data.get('comercio', 'Desconocido'),
                'monto': self._validate_amount(data.get('monto')),
                'metodo_pago': data.get('metodo_pago'),
                'categoria': self._validate_category(data.get('categoria')),
            }
            
            # Logging
            print(f"  [+] Comercio: {expense['comercio']}")
            print(f"  [+] Monto: ${expense['monto']:.2f}" if expense['monto'] else "  [+] Monto: N/A")
            print(f"  [+] Fecha: {expense['fecha']}" if expense['fecha'] else "  [+] Fecha: N/A")
            print(f"  [+] Pago: {expense['metodo_pago']}" if expense['metodo_pago'] else "  [+] Pago: N/A")
            print(f"  [+] Categoria: {expense['categoria']}")
            
            return expense
            
        except requests.exceptions.Timeout:
            print(f"  [!] Timeout - La IA tardo demasiado")
            return self._create_error_result(image_path.name, "Timeout")
        except Exception as e:
            print(f"  [!] Error inesperado: {e}")
            return self._create_error_result(image_path.name, str(e))
    
    def _validate_date(self, date_str):
        """Valida y normaliza fecha"""
        if not date_str:
            return None
        
        # Si ya está en formato correcto
        if isinstance(date_str, str) and len(date_str) == 10 and date_str.count('-') == 2:
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                return date_str
            except:
                pass
        
        # Intentar parsear otros formatos
        formats = ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']
        for fmt in formats:
            try:
                dt = datetime.strptime(str(date_str), fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
        
        return None
    
    def _validate_amount(self, amount):
        """Valida y limpia monto"""
        if amount is None:
            return None
        
        try:
            # Si es string, limpiar
            if isinstance(amount, str):
                amount = amount.replace('$', '').replace(',', '').strip()
            
            amount = float(amount)
            
            # Validar rango razonable
            if 0.01 <= amount <= 1000000:
                return amount
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _validate_category(self, category):
        """Valida que la categoría sea válida"""
        if not category:
            return 'Otros'
        
        valid_categories = list(self.categories.keys())
        
        # Buscar coincidencia exacta
        if category in valid_categories:
            return category
        
        # Buscar coincidencia case-insensitive
        category_lower = category.lower()
        for valid_cat in valid_categories:
            if valid_cat.lower() == category_lower:
                return valid_cat
        
        return 'Otros'
    
    def _create_error_result(self, filename, error_msg):
        """Crea resultado de error"""
        return {
            'archivo': filename,
            'fecha': None,
            'hora': None,
            'comercio': f'Error: {error_msg}',
            'monto': None,
            'metodo_pago': None,
            'categoria': 'Error'
        }
    
    def process_folder(self, folder_path):
        """Procesa todos los tickets en una carpeta"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"[!] Error: La carpeta {folder_path} no existe")
            return None
        
        # Buscar imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(folder.glob(f'*{ext}'))
        
        if not images:
            print(f"[!] No se encontraron imagenes en {folder_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Encontradas {len(images)} imagenes")
        print(f"{'='*60}\n")
        
        expenses = []
        for i, img_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}]")
            expense = self.analyze_receipt_with_ai(img_path)
            expenses.append(expense)
        
        return pd.DataFrame(expenses)
    
    def generate_report(self, df, output_file='expenses_report_AI.csv'):
        """Genera reporte con estadísticas"""
        if df is None or df.empty:
            print("\n[!] No hay datos para generar reporte")
            return
        
        # Guardar CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n{'='*60}")
        print(f"[+] Reporte guardado en: {output_file}")
        print(f"{'='*60}")
        
        # Estadísticas
        print("\n" + "="*60)
        print("RESUMEN DE GASTOS")
        print("="*60)
        
        valid_amounts = df[df['monto'].notna()]
        total = valid_amounts['monto'].sum()
        
        print(f"\nTotal de gastos: ${total:,.2f}")
        print(f"Tickets procesados: {len(df)}")
        print(f"Tickets con monto: {len(valid_amounts)} ({len(valid_amounts)/len(df)*100:.1f}%)")
        print(f"Tickets con fecha: {df['fecha'].notna().sum()} ({df['fecha'].notna().sum()/len(df)*100:.1f}%)")
        print(f"Tickets con metodo de pago: {df['metodo_pago'].notna().sum()} ({df['metodo_pago'].notna().sum()/len(df)*100:.1f}%)")
        
        if len(valid_amounts) > 0:
            print("\n" + "-"*60)
            print("GASTOS POR CATEGORIA")
            print("-"*60)
            by_category = valid_amounts.groupby('categoria')['monto'].agg(['sum', 'count', 'mean'])
            by_category = by_category.sort_values('sum', ascending=False)
            for cat, row in by_category.iterrows():
                percentage = (row['sum'] / total) * 100
                print(f"{cat:20s}: ${row['sum']:8,.2f} ({percentage:5.1f}%) - {int(row['count'])} tickets - Promedio: ${row['mean']:,.2f}")
            
            print("\n" + "-"*60)
            print("TOP 5 COMERCIOS")
            print("-"*60)
            by_merchant = valid_amounts.groupby('comercio')['monto'].agg(['sum', 'count'])
            by_merchant = by_merchant.sort_values('sum', ascending=False).head(5)
            for i, (merchant, row) in enumerate(by_merchant.iterrows(), 1):
                print(f"{i}. {merchant:30s}: ${row['sum']:8,.2f} ({int(row['count'])} tickets)")
            
            print("\n" + "-"*60)
            print("ESTADISTICAS")
            print("-"*60)
            avg = valid_amounts['monto'].mean()
            median = valid_amounts['monto'].median()
            max_expense = valid_amounts['monto'].max()
            min_expense = valid_amounts['monto'].min()
            print(f"Promedio por ticket: ${avg:,.2f}")
            print(f"Mediana: ${median:,.2f}")
            print(f"Gasto maximo: ${max_expense:,.2f}")
            print(f"Gasto minimo: ${min_expense:,.2f}")
        
        return df


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("EXPENSE TRACKER - Analisis con IA Local (Ollama)")
    print("="*60)
    
    # Configuración
    # Puedes cambiar el modelo aquí según tu hardware:
    # - 'llama3:latest' (recomendado, 16GB RAM)
    # - 'llava:7b' (más rápido, 8GB RAM)
    # - 'llava:13b' (balance, 16GB RAM)
    
    model = 'llava:7b'
    try:
        tracker = AIExpenseTracker(model=model)
        folder_path = './data'
        
        # Procesar tickets
        df = tracker.process_folder(folder_path)
        
        # Generar reporte
        if df is not None:
            output_file = 'expenses_report_AI.csv'
            tracker.generate_report(df, output_file)
            
            print("\n" + "="*60)
            print("PROCESO COMPLETADO")
            print("="*60)
            print(f"\nArchivo generado: {output_file}")
            print("Tip: Compara con el CSV del OCR tradicional para ver mejoras")
            print()
    
    except Exception as e:
        print(f"\n[!] Error fatal: {e}")
        print("\nVerifica que:")
        print("1. Ollama este corriendo (ollama serve)")
        print(f"2. El modelo este instalado (ollama pull {model})")
        print("3. Tengas suficiente RAM/VRAM")


if __name__ == "__main__":
    main()