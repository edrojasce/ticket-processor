#!/usr/bin/env python3
"""
Expense Tracker - Análisis de Tickets con OCR Local
Usa Tesseract OCR para extraer texto de imágenes de tickets
100% local y rápido

Instalación:
pip install pytesseract pillow pandas opencv-python

También necesitas instalar Tesseract OCR:
- macOS: brew install tesseract tesseract-lang
- Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-spa
- Windows: descargar de https://github.com/UB-Mannheim/tesseract/wiki
"""

import re
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from PIL import Image
import pytesseract
import cv2
import numpy as np
try:
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.utils.dataframe import dataframe_to_rows
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("[!] ADVERTENCIA: openpyxl no está instalado")
    print("    Para Excel con imágenes: pip install openpyxl")

class ExpenseTracker:
    def __init__(self):
        self.categories = {
            'Alimentos': ['oxxo', 'seven', '7-eleven', 'eleven', 'walmart', 'soriana', 'chedraui', 
                         'superama', 'bodega', 'aurrera', 'costco', 'sams', 'heb', 'super',
                         'restaurant', 'cafe', 'coffee', 'starbucks', 'subway', 'wings',
                         'mcdonalds', 'burger', 'pizza', 'taco', 'comida', 'food', 'kitchen',
                         'bistro', 'grill', 'diner', 'cantina', 'salad', 'chicken', 'papos',
                         'clean papos'],
            'Transporte': ['uber', 'didi', 'gas', 'gasolina', 'pemex', 'shell', 
                          'mobil', 'bp', 'taxi', 'parking', 'estacionamiento'],
            'Entretenimiento': ['cinema', 'cine', 'cinepolis', 'cinemex', 'netflix',
                               'spotify', 'xbox', 'playstation', 'steam', 'hobby', 'lobby'],
            'Salud': ['farmacia', 'pharmacy', 'guadalajara', 'ahorro', 'similares',
                     'benavides', 'hospital', 'doctor', 'clinic', 'medic'],
            'Servicios': ['electric', 'cfe', 'agua', 'telmex', 'telcel', 'att',
                         'movistar', 'izzi', 'totalplay', 'dish', 'netpay', 'steren'],
            'Ropa': ['zara', 'h&m', 'liverpool', 'palacio', 'coppel', 'suburbia',
                    'nike', 'adidas', 'clothing', 'shoes'],
            'Tecnologia': ['best buy', 'office', 'depot', 'steren', 'radioshack',
                          'apple', 'samsung', 'dell', 'hp'],
            'Otros': []
        }
        
        self.patterns = {
            'amount': [
                # Patrones para "Total" con diferentes formatos
                r'total\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'total\s*a?\s*pagar\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'importe\s*total\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'neto\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                # Monto después de "Total:" en la misma o siguiente línea
                r'total\s*:?\s*[\r\n]+\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                # Patrones con $ explícito
                r'\$\s*(\d{1,3}(?:[,\.]\d{3})*\.\d{2})\b',
            ],
            'date': [
                # Fecha con formato dd/mm/yyyy o dd-mm-yyyy
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2})\b',
                # Fecha con palabra clave
                r'fecha\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                # Formato con espacios o puntos
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
                r'card\s*(?:no\.?)?\s*:?\s*\*+\s*(\d{4})',
                r'tc\s*:?\s*\*+\s*(\d{4})',
                # Buscar 4 dígitos después de asteriscos
                r'\*+\s*(\d{4})(?:\s|$)',
            ]
        }
    
    def preprocess_image_multiple_methods(self, image_path):
        """Aplica múltiples técnicas de preprocesamiento"""
        img = cv2.imread(str(image_path))
        
        if img is None:
            return []
        
        # Redimensionar si es muy grande (para acelerar)
        max_dimension = 2000
        height, width = img.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_images = []
        
        # Método 1: Escala de grises + contraste mejorado
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(('enhanced', enhanced))
        
        # Método 2: Umbral adaptativo
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(('adaptive', adaptive))
        
        # Método 3: Otsu + denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('otsu', otsu))
        
        # Método 4: Aumentar tamaño 2x (mejor OCR para texto pequeño)
        height, width = gray.shape
        scaled = cv2.resize(gray, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        processed_images.append(('scaled', scaled))
        
        return processed_images
    
    def extract_text_multiple_configs(self, image_path):
        """Extrae texto con múltiples configuraciones de Tesseract"""
        processed_images = self.preprocess_image_multiple_methods(image_path)
        
        if not processed_images:
            return ""
        
        all_texts = []
        
        # Configuraciones optimizadas de Tesseract
        configs = [
            '--oem 3 --psm 6 -l spa+eng',  # Bloque de texto uniforme
            '--oem 3 --psm 4 -l spa+eng',  # Columna única de texto
            '--oem 3 --psm 11 -l spa+eng', # Texto disperso sin orden
        ]
        
        # Probar diferentes combinaciones (optimizado)
        for method_name, processed_img in processed_images:
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text and len(text.strip()) > 20:
                        all_texts.append(text.lower())
                except Exception as e:
                    continue
        
        # Combinar todos los textos
        combined_text = '\n'.join(all_texts)
        return combined_text
    
    def extract_amount(self, text):
        """Extrae el monto del ticket con prioridad al total"""
        amounts = []
        total_amounts = []
        
        # Buscar específicamente líneas con "Total"
        lines = text.split('\n')
        for line in lines:
            if 'total' in line.lower():
                # Extraer números de esta línea
                numbers = re.findall(r'(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)', line)
                for num in numbers:
                    try:
                        clean = num.replace(',', '')
                        amount = float(clean)
                        if 0.01 <= amount <= 100000:
                            total_amounts.append(amount)
                    except ValueError:
                        continue
        
        # Si encontramos montos cerca de "total", usar el mayor
        if total_amounts:
            return max(total_amounts)
        
        # Buscar todos los montos con los patrones
        for pattern in self.patterns['amount']:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                clean_amount = match.replace(',', '').replace(' ', '')
                try:
                    amount = float(clean_amount)
                    if 0.01 <= amount <= 100000:
                        amounts.append(amount)
                except ValueError:
                    continue
        
        if not amounts:
            return None
        
        # Usar el monto más grande (normalmente es el total)
        return max(amounts)
    
    def extract_date(self, text):
        """Extrae la fecha del ticket"""
        # Buscar líneas que contengan "fecha" primero
        lines = text.split('\n')
        for line in lines:
            if 'fecha' in line.lower() or 'date' in line.lower():
                date_matches = re.findall(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', line)
                if date_matches:
                    date_str = date_matches[0]
                    parsed_date = self.parse_date(date_str)
                    if parsed_date:
                        return parsed_date
        
        # Buscar en todo el texto
        for pattern in self.patterns['date']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                parsed_date = self.parse_date(match.strip())
                if parsed_date:
                    return parsed_date
        
        return None
    
    def parse_date(self, date_str):
        """Intenta parsear una fecha con múltiples formatos"""
        date_str = date_str.replace('.', '/').replace(' ', '/')
        
        formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d',
            '%d/%m/%y', '%d-%m-%y', '%m/%d/%Y', '%m-%d-%Y',
        ]
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                # Ajustar año si es de 2 dígitos
                if date_obj.year < 100:
                    date_obj = date_obj.replace(year=date_obj.year + 2000)
                # Validar rango razonable
                if 2020 <= date_obj.year <= 2030:
                    return date_obj.strftime('%Y-%m-%d')
            except (ValueError, AttributeError):
                continue
        
        return None
    
    def extract_time(self, text):
        """Extrae la hora del ticket"""
        for pattern in self.patterns['time']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def extract_card_last4(self, text):
        """Extrae últimos 4 dígitos de tarjeta o detecta efectivo"""
        # Detectar efectivo
        if re.search(r'\befectivo\b|\bcash\b|\bcontado\b', text, re.IGNORECASE):
            return 'Efectivo'
        
        # Buscar patrones de tarjeta
        for pattern in self.patterns['card']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                digits = matches[0]
                if isinstance(digits, str) and digits.isdigit() and len(digits) == 4:
                    return f"****{digits}"
        
        # Detectar tipos de tarjeta sin dígitos
        if re.search(r'\bvisa\b', text, re.IGNORECASE):
            return 'Visa'
        if re.search(r'\bmastercard\b', text, re.IGNORECASE):
            return 'Mastercard'
        if re.search(r'\bamex\b|\bamerican express\b', text, re.IGNORECASE):
            return 'Amex'
        
        return None
    
    def extract_merchant(self, text):
        """Extrae el nombre del comercio de las primeras líneas"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        candidates = []
        for i, line in enumerate(lines[:15]):  # Revisar más líneas
            # Filtrar líneas válidas para nombre de comercio
            if (3 <= len(line) <= 60 and 
                not re.match(r'^[\d\s:\-/\.\$]+$', line) and
                not re.search(r'\d{5,}', line) and  # No tenga números largos
                not line.startswith(('http', 'www', '@'))):  # No sea URL
                
                # Dar más peso a las primeras líneas
                score = 15 - i
                
                # Bonus si tiene palabras como restaurant, cafe, etc
                if any(word in line.lower() for word in ['restaurant', 'cafe', 'store', 'shop', 'market']):
                    score += 5
                
                candidates.append((line, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            merchant = candidates[0][0]
            # Limpiar y formatear
            merchant = re.sub(r'[^\w\s\-\.]', '', merchant)
            merchant = ' '.join(merchant.split())  # Normalizar espacios
            return merchant.title()[:50]
        
        return 'Desconocido'
    
    def categorize_expense(self, text, merchant):
        """Categoriza el gasto basado en palabras clave"""
        text_combined = f"{text} {merchant}".lower()
        
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(text_combined.count(keyword) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'Otros'
    
    def _generate_excel_with_images(self, df, excel_file, image_folder):
        """Genera Excel con imágenes embebidas de cada ticket"""
        if not EXCEL_AVAILABLE:
            print("[!] No se puede generar Excel con imágenes (instala openpyxl)")
            return
        
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Gastos con Imágenes"
            
            # Headers
            headers = ['Imagen', 'Comercio', 'Fecha', 'Hora', 'Monto', 'Método Pago', 'Categoría', 'Archivo']
            ws.append(headers)
            
            # Estilo de headers
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(color="FFFFFF", bold=True)
            
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            
            # Configurar anchos de columna
            ws.column_dimensions['A'].width = 30  # Imagen
            ws.column_dimensions['B'].width = 25  # Comercio
            ws.column_dimensions['C'].width = 12  # Fecha
            ws.column_dimensions['D'].width = 10  # Hora
            ws.column_dimensions['E'].width = 12  # Monto
            ws.column_dimensions['F'].width = 15  # Método Pago
            ws.column_dimensions['G'].width = 15  # Categoría
            ws.column_dimensions['H'].width = 35  # Archivo
            
            # Procesar cada fila
            image_folder_path = Path(image_folder)
            row_idx = 2  # Empezar después del header
            
            for idx, row in df.iterrows():
                # Insertar datos
                ws.cell(row=row_idx, column=2, value=row['comercio'])
                ws.cell(row=row_idx, column=3, value=row['fecha'])
                ws.cell(row=row_idx, column=4, value=row['hora'])
                ws.cell(row=row_idx, column=5, value=f"${row['monto']:,.2f}" if pd.notna(row['monto']) else "N/A")
                ws.cell(row=row_idx, column=6, value=row['metodo_pago'])
                ws.cell(row=row_idx, column=7, value=row['categoria'])
                ws.cell(row=row_idx, column=8, value=row['archivo'])
                
                # Centrar texto
                for col in range(2, 9):
                    ws.cell(row=row_idx, column=col).alignment = Alignment(horizontal="center", vertical="center")
                
                # Insertar imagen
                image_path = image_folder_path / row['archivo']
                if image_path.exists():
                    try:
                        # Crear thumbnail de la imagen
                        img = Image.open(image_path)
                        
                        # Redimensionar para que quepa bien en Excel
                        max_height = 200
                        ratio = max_height / img.height
                        new_size = (int(img.width * ratio), max_height)
                        img.thumbnail(new_size, Image.Resampling.LANCZOS)
                        
                        # Guardar thumbnail temporal
                        temp_path = Path(f"temp_thumb_{idx}.jpg")
                        img.save(temp_path, "JPEG")
                        
                        # Insertar en Excel
                        xl_img = XLImage(str(temp_path))
                        xl_img.anchor = f'A{row_idx}'
                        ws.add_image(xl_img)
                        
                        # Ajustar altura de fila
                        ws.row_dimensions[row_idx].height = 150
                        
                        # Limpiar temporal
                        temp_path.unlink()
                        
                    except Exception as e:
                        print(f"  [!] Error al insertar imagen {row['archivo']}: {e}")
                        ws.cell(row=row_idx, column=1, value="Error cargando imagen")
                else:
                    ws.cell(row=row_idx, column=1, value="Imagen no encontrada")
                
                row_idx += 1
            
            # Guardar Excel
            wb.save(excel_file)
            print(f"[+] Excel generado exitosamente con {len(df)} tickets")
            
        except Exception as e:
            print(f"[!] Error al generar Excel: {e}")
    
    def analyze_receipt(self, image_path):
        """Analiza un ticket completo"""
        print(f"Procesando: {image_path.name}")
        
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
        
        # Extraer información
        merchant = self.extract_merchant(text)
        amount = self.extract_amount(text)
        date = self.extract_date(text)
        time = self.extract_time(text)
        payment = self.extract_card_last4(text)
        category = self.categorize_expense(text, merchant)
        
        # Logging de resultados
        print(f"  [+] Comercio: {merchant}")
        print(f"  [+] Monto: ${amount:.2f}" if amount else "  [+] Monto: N/A")
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
            'texto_extraido': text[:500]  # Más texto para debug
        }
    
    def process_folder(self, folder_path):
        """Procesa todos los tickets en una carpeta"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: La carpeta {folder_path} no existe")
            return None
        
        # Buscar imágenes solo en la carpeta principal (no subcarpetas)
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
            print()  # Línea en blanco
        
        return pd.DataFrame(expenses)
    
    def generate_report(self, df, output_file='expenses_report.csv'):
        """Genera reporte con estadísticas"""
        if df is None or df.empty:
            print("No hay datos para generar reporte")
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
    print("EXPENSE TRACKER - Analisis de Tickets con OCR")
    print("="*60)
    
    tracker = ExpenseTracker()
    folder_path = './data'
    
    # Procesar tickets
    df = tracker.process_folder(folder_path)
    
    # Generar reporte
    if df is not None:
        output_file = 'expenses_report.csv'
        tracker.generate_report(df, output_file, folder_path)
        
        print("\n" + "="*60)
        print("PROCESO COMPLETADO")
        print("="*60)
        print(f"\nArchivos generados:")
        print(f"  - {output_file} (CSV para Excel/Sheets)")
        print(f"  - expenses_report_con_imagenes.xlsx (Excel con fotos)")
        print("\nTip: Abre el archivo XLSX para revisar visualmente cada ticket")
        print()


if __name__ == "__main__":
    main()