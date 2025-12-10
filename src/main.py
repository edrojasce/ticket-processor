import re
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import pytesseract
import cv2
import numpy as np

class ExpenseTracker:
    def __init__(self):
        # 1. CATEGORÍAS AMPLIADAS Y MEJORADAS
        self.categories = {
            'Alimentos': ['oxxo', 'seven', '7-eleven', 'eleven', 'walmart', 'soriana', 'chedraui', 
                         'superama', 'bodega', 'aurrera', 'costco', 'sams', 'heb', 'super',
                         'restaurant', 'cafe', 'coffee', 'starbucks', 'subway', 'wings',
                         'mcdonalds', 'burger', 'pizza', 'taco', 'comida', 'food', 'kitchen',
                         'bistro', 'grill', 'diner', 'cantina', 'salad', 'chicken', 'bar', 'tienda', 'abarrotes'],
            'Transporte': ['uber', 'didi', 'gas', 'gasolina', 'pemex', 'shell', 
                          'mobil', 'bp', 'taxi', 'parking', 'estacionamiento', 'peaje', 'autobus', 'metro'],
            'Entretenimiento': ['cinema', 'cine', 'cinepolis', 'cinemex', 'netflix',
                               'spotify', 'xbox', 'playstation', 'steam', 'hobby', 'lobby', 'teatro', 'concierto'],
            'Salud': ['farmacia', 'pharmacy', 'guadalajara', 'ahorro', 'similares',
                     'benavides', 'hospital', 'doctor', 'clinic', 'medic', 'analisis', 'laboratorio', 'dentista'],
            'Servicios': ['electric', 'cfe', 'agua', 'telmex', 'telcel', 'att',
                         'movistar', 'izzi', 'totalplay', 'dish', 'netpay', 'clean', 'lavanderia', 'internet', 'renta'],
            'Ropa': ['zara', 'h&m', 'liverpool', 'palacio', 'coppel', 'suburbia',
                    'nike', 'adidas', 'clothing', 'shoes', 'boutique', 'fashion'],
            'Tecnologia': ['best buy', 'office', 'depot', 'steren', 'radioshack',
                          'apple', 'samsung', 'dell', 'hp', 'electronica', 'computo'],
            'Hogar': ['ferreteria', 'plomeria', 'muebles', 'home depot', 'sodimac', 'tlapaleria', 'materiales'],
            'Viajes': ['aeropuerto', 'vuelo', 'hotel', 'airbnb', 'booking', 'turismo', 'avion', 'boletos'],
            'Otros': []
        }
        
        # 2. PATRONES DE EXTRACCIÓN MEJORADOS (Más flexibles para el monto)
        self.patterns = {
            'amount': [
                # Patrones robustos para "Total" con flexibilidad en $ y separadores (., o espacio)
                r'(?:total|importe|neto|suma|pago|cambio|a pagar|total a pagar)\s*:?\s*[\$S]?\s*(\d{1,3}(?:[,\s\.]\d{3})*(?:[,\.]\d{2}))',
                # Patrones con $ explícito (priorizando 2 decimales)
                r'\$\s*(\d{1,3}(?:[,\s\.]\d{3})*[,\.]\d{2})\b',
            ],
            'date': [
                # Formatos comunes (dd/mm/yyyy, yyyy/mm/dd, etc.)
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2})\b',
                # Con separadores de punto o espacio
                r'\b(\d{1,2}[.\s]\d{1,2}[.\s]\d{2,4})\b',
            ],
            'time': [
                # Hora con formato HH:MM o HH:MM:SS
                r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b',
                r'hora\s*:?\s*(\d{1,2}:\d{2})',
            ],
            'card': [
                # Patrones para últimos 4 dígitos de tarjeta
                r'\*{4,}\s*(\d{4})\b',
                r'xxxx\s*(\d{4})\b',
                r'tarjeta\s*(?:no\.?)?\s*:?\s*\*+\s*(\d{4})',
                r'tc\s*:?\s*\*+\s*(\d{4})',
            ]
        }
        # Configurar Tesseract (asegúrate de que esté en el PATH o configúralo aquí)
        # pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'
    
    def preprocess_image_multiple_methods(self, image_path):
        """Aplica múltiples técnicas de preprocesamiento, incluyendo escalado y umbralización inversa."""
        img = cv2.imread(str(image_path))
        
        if img is None:
            return []
        
        max_dimension = 2500 # Aumentar un poco el límite
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_images = []
        
        # 1. Escala de grises + contraste mejorado
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(('enhanced', enhanced))
        
        # 2. Umbral adaptativo
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(('adaptive', adaptive))
        
        # 3. Otsu (con denoise)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('otsu', otsu))
        
        # 4. Umbralización Inversa (para texto claro sobre fondo oscuro)
        _, otsu_inv = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        processed_images.append(('otsu_inv', otsu_inv)) # ¡Nuevo!
        
        # 5. Aumentar tamaño 2x (mejor OCR para texto pequeño)
        height, width = gray.shape
        scaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        processed_images.append(('scaled', scaled))
        
        return processed_images
    
    def extract_text_multiple_configs(self, image_path):
        """Extrae texto con múltiples configuraciones y prioriza el texto de mejor calidad."""
        processed_images = self.preprocess_image_multiple_methods(image_path)
        
        if not processed_images:
            return ""
        
        candidate_texts = []
        
        # Configuraciones optimizadas de Tesseract
        configs = [
            '--oem 3 --psm 6 -l spa+eng',  # Bloque de texto uniforme (Mejor para tickets)
            '--oem 3 --psm 4 -l spa+eng',  # Columna única de texto
            '--oem 3 --psm 11 -l spa+eng', # Texto disperso sin orden
        ]
        
        # Palabras clave para puntuar la calidad del OCR
        keywords = self.patterns['amount'] + self.patterns['date'] + ['total', 'fecha', 'importe', 'pago', 'caja', 'folio']
        
        for method_name, processed_img in processed_images:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text and len(text.strip()) > 20:
                        text_lower = text.lower()
                        # Lógica de Puntuación: contar cuántas palabras clave encuentra
                        score = sum(text_lower.count(re.escape(k)) for k in keywords) 
                        
                        candidate_texts.append({'text': text_lower, 'score': score})
                except Exception as e:
                    continue
        
        if not candidate_texts:
            return ""
            
        # Priorizar el texto con la mayor puntuación (mayor densidad de información útil)
        candidate_texts.sort(key=lambda x: x['score'], reverse=True)
        
        # Si la mejor puntuación es muy baja (ej. 0), simplemente devolvemos la más larga
        if candidate_texts[0]['score'] < 3:
            return max(candidate_texts, key=lambda x: len(x['text']))['text']
            
        return candidate_texts[0]['text']
    
    def clean_amount(self, amount_str):
        """Limpia y normaliza un string de monto, manejando comas/puntos decimales."""
        amount_str = amount_str.strip().replace('$', '').replace('S', '')
        
        # Paso 1: Normalizar separadores de miles/decimales
        if amount_str.count(',') > 1 and amount_str.count('.') == 1:
            # Caso: 1,234,567.89 (Coma de miles, punto decimal)
            cleaned = amount_str.replace(',', '')
        elif amount_str.count('.') > 1 and amount_str.count(',') == 1:
            # Caso: 1.234.567,89 (Punto de miles, coma decimal)
            cleaned = amount_str.replace('.', '').replace(',', '.')
        elif amount_str.count('.') > 1:
            # Caso: 1.234.56 (Asume que el último punto es decimal, ej. 1.234,56)
            cleaned = amount_str[:amount_str.rfind('.')] + amount_str[amount_str.rfind('.'):].replace('.', '')
        elif amount_str.count(',') == 1 and len(amount_str.split(',')[-1]) == 2:
            # Caso: 1,99 (Asume coma decimal si solo hay 2 dígitos después)
            cleaned = amount_str.replace(',', '.')
        else:
            cleaned = amount_str.replace(',', '') # Quitar comas (asumiendo miles)
        
        try:
            amount = float(cleaned)
            if 0.01 <= amount <= 100000:
                return amount
        except ValueError:
            pass
        return None
    
    def extract_amount(self, text):
        """Extrae el monto del ticket con prioridad al total y aplicando limpieza robusta."""
        amounts = []
        
        # 1. Búsqueda con patrones explícitos de "Total" (más robustos)
        for pattern in self.patterns['amount']:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                clean_amount = self.clean_amount(match)
                if clean_amount is not None:
                    amounts.append(clean_amount)
        
        # Si ya encontramos montos asociados a 'total', usamos el mayor (la heurística más fuerte)
        if amounts:
            return max(amounts)

        # 2. Búsqueda de cualquier número que parezca un monto (ej. con dos decimales)
        potential_amounts = re.findall(r'(\d{1,3}(?:[,\s\.]\d{3})*[,\.]\d{2})\s*$', text, re.IGNORECASE | re.MULTILINE)
        
        for pa in potential_amounts:
             clean_amount = self.clean_amount(pa)
             if clean_amount is not None:
                amounts.append(clean_amount)
        
        if not amounts:
            return None
        
        # Usar el monto más grande como fallback (normalmente es el total)
        return max(amounts)
    
    def parse_date(self, date_str):
        """Intenta parsear una fecha con múltiples formatos y lógica de D/M/Y."""
        date_str = date_str.replace('.', '/').replace(' ', '/').replace('-', '/')
        
        # Formatos comunes (prioridad a D/M/Y ya que muchos tickets lo usan)
        formats = [
            '%d/%m/%Y', '%d/%m/%y', # D/M/Y (D/M/YY)
            '%m/%d/%Y', '%m/%d/%y', # M/D/Y (M/D/YY)
            '%Y/%m/%d', '%y/%m/%d', # Y/M/D (YY/M/D)
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                # Lógica de D/M/Y: Si el formato es ambiguo (D/M vs M/D) y el primer número > 12, priorizamos que sea el día.
                parts = date_str.split('/')
                if len(parts) == 3 and (fmt == '%d/%m/%Y' or fmt == '%d/%m/%y') and int(parts[0]) > 12:
                    # Si el día es > 12, es muy probable que sea D/M/Y
                    pass 
                elif len(parts) == 3 and (fmt == '%m/%d/%Y' or fmt == '%m/%d/%y') and int(parts[0]) > 12:
                    # Si el mes es > 12, esto es un error (se salta)
                    continue
                
                # Ajustar año si es de 2 dígitos
                if date_obj.year < 100:
                    date_obj = date_obj.replace(year=date_obj.year + 2000)
                # Validar rango razonable
                if datetime.now().year - 5 <= date_obj.year <= datetime.now().year + 1:
                    return date_obj.strftime('%Y-%m-%d')
            except (ValueError, AttributeError):
                continue
        
        return None
    
    def extract_date(self, text):
        """Extrae la fecha, priorizando cercanía a 'fecha' y 'hora'."""
        
        # Buscar en todo el texto (más flexible)
        for pattern in self.patterns['date']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                parsed_date = self.parse_date(match.strip())
                if parsed_date:
                    return parsed_date
        
        return None
    
    def extract_time(self, text):
        """Extrae la hora del ticket."""
        # Se mantiene la lógica simple, es menos problemática
        for pattern in self.patterns['time']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def extract_card_last4(self, text):
        """Extrae últimos 4 dígitos de tarjeta o detecta efectivo."""
        # Detectar efectivo
        if re.search(r'\befectivo\b|\bcash\b|\bcontado\b|\bmoneda nacional\b', text, re.IGNORECASE):
            return 'Efectivo'
        
        # Buscar patrones de tarjeta
        for pattern in self.patterns['card']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                digits = matches[0]
                if isinstance(digits, str) and digits.isdigit() and len(digits) == 4:
                    # Buscar la palabra 'Tarjeta' cerca (para evitar números al azar)
                    context_match = re.search(r'(?:tarjeta|card|pago|tc)\s*(?:no\.?)?\s*[*xX]{0,}\s*' + re.escape(digits), text, re.IGNORECASE)
                    if context_match:
                         return f"****{digits}"
        
        # Detección de tipos de tarjeta sin dígitos (como fallback)
        if re.search(r'\bvisa\b', text, re.IGNORECASE):
            return 'Visa'
        if re.search(r'\bmastercard\b', text, re.IGNORECASE):
            return 'Mastercard'
        
        return None
    
    def extract_merchant(self, text):
        """Extrae el nombre del comercio de las primeras líneas, filtrando ruido."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        candidates = []
        # Revisar solo las primeras 10 líneas, donde suele estar el nombre
        for i, line in enumerate(lines[:10]): 
            line_lower = line.lower()
            
            # FILTRADO DE RUIDO (¡Mejorado!)
            is_noise = any(keyword in line_lower for keyword in [
                'c.p.', 'tel', 'rfc', 'calle', 'ave', 'avenida', 'direccion', 
                'factura', 'ticket', 'folio', 'caja', 'hora', 'fecha', 'impuesto', 'iva',
                'subtotal', 'total'
            ])
            
            is_url_or_large_number = (re.search(r'\d{5,}', line) or 
                                      line.startswith(('http', 'www', '@', '+52', '55')))
            
            # Criterio: Longitud razonable, no solo números/símbolos, y no es ruido de dirección/facturación
            if (3 <= len(line) <= 60 and 
                not re.match(r'^[\d\s:\-/\.\$]+$', line) and
                not is_url_or_large_number and
                not is_noise):
                
                # Puntuación basada en la posición
                score = 10 - i 
                
                # Bonus si contiene palabras clave que denotan un comercio
                if any(word in line_lower for word in ['store', 'shop', 'market', 'cafe', 'restaurant']):
                    score += 5
                
                candidates.append((line, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            merchant = candidates[0][0]
            # Limpiar y formatear
            merchant = re.sub(r'[^\w\s\-\.]', '', merchant)
            merchant = ' '.join(merchant.split())
            return merchant.title()[:50]
        
        return 'Desconocido'
    
    def categorize_expense(self, text, merchant):
        """Categoriza el gasto basado en palabras clave."""
        text_combined = f"{text} {merchant}".lower()
        
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(text_combined.count(keyword) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Devuelve la categoría con la puntuación más alta
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'Otros'
    
    def analyze_receipt(self, image_path):
        """Analiza un ticket completo"""
        print(f"Procesando: {image_path.name}")
        
        # USANDO EL NUEVO MÉTODO DE EXTRACCIÓN CON PUNTUACIÓN
        text = self.extract_text_multiple_configs(image_path)
        
        if not text or len(text.strip()) < 20:
            print(f"  [!] Advertencia: Texto extraido muy corto o vacio")
            return {
                'archivo': image_path.name,
                'fecha': None,
                'hora': None,
                'comercio': 'Error OCR',
                'monto': None,
                'metodo_pago': None,
                'categoria': 'Error',
                'texto_extraido': text[:300] if text else ''
            }
        
        # Extraer información con las funciones mejoradas
        merchant = self.extract_merchant(text)
        amount = self.extract_amount(text)
        date = self.extract_date(text)
        time = self.extract_time(text)
        payment = self.extract_card_last4(text)
        category = self.categorize_expense(text, merchant)
        
        # Logging de resultados
        print(f"  [+] Comercio: {merchant}")
        print(f"  [+] Monto: ${amount:,.2f}" if amount else "  [+] Monto: N/A")
        print(f"  [+] Fecha: {date}" if date else "  [+] Fecha: N/A")
        print(f"  [+] Pago: {payment}" if payment else "  [+] Pago: N/A")
        print(f"  [+] Categoria: {category}")
        
        return {
            'archivo': image_path.name,
            'fecha': date,
            'hora': time,
            'comercio': merchant,
            'monto': amount,
            'metodo_pago': payment,
            'categoria': category,
            'texto_extraido': text[:500]
        }
    
    # El resto de los métodos (process_folder, generate_report) se mantienen igual
    
    def process_folder(self, folder_path):
        """Procesa todos los tickets en una carpeta"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: La carpeta {folder_path} no existe")
            return None
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(folder.glob(f'*{ext}'))
        
        if not images:
            print(f"No se encontraron imagenes en {folder_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Encontradas {len(images)} imagenes")
        print(f"{'='*60}\n")
        
        expenses = []
        for img_path in images:
            expense = self.analyze_receipt(img_path)
            expenses.append(expense)
            print()
        
        return pd.DataFrame(expenses)
    
    def generate_report(self, df, output_file='expenses_report.csv'):
        """Genera reporte con estadísticas"""
        if df is None or df.empty:
            print("No hay datos para generar reporte")
            return
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n{'='*60}")
        print(f"[+] Reporte guardado en: {output_file}")
        print(f"{'='*60}")
        
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
    print("EXPENSE TRACKER - Analisis de Tickets con OCR")
    print("="*60)
    
    tracker = ExpenseTracker()
    # Asegúrate de que tu carpeta 'data' exista y contenga imágenes de tickets
    folder_path = './data' 
    
    if not Path(folder_path).is_dir():
        print(f"CREANDO CARPETA DE DATOS: '{folder_path}'")
        Path(folder_path).mkdir(exist_ok=True)
        print("¡Coloca tus imágenes de tickets dentro de esta carpeta!")

    # Procesar tickets
    df = tracker.process_folder(folder_path)
    
    # Generar reporte
    if df is not None:
        output_file = 'expenses_report.csv'
        tracker.generate_report(df, output_file)
        
        print("\n" + "="*60)
        print("PROCESO COMPLETADO")
        print("="*60)
        print(f"\nArchivo generado: {output_file}")
        print("Tip: Abre el CSV en Excel o Google Sheets para analisis detallado")
        print()


if __name__ == "__main__":
    main()