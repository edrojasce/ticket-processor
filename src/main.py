#!/usr/bin/env python3
"""
Expense Tracker - Análisis de Tickets con OCR Local Mejorado
Versión robusta con múltiples estrategias de extracción

Instalación:
pip install pytesseract pillow pandas opencv-python numpy

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

class ExpenseTracker:
    def __init__(self):
        self.categories = {
            'Alimentos': ['oxxo', 'seven', '7-eleven', 'eleven', 'walmart', 'soriana', 'chedraui', 
                         'superama', 'bodega', 'aurrera', 'costco', 'sams', 'heb', 'super',
                         'restaurant', 'cafe', 'coffee', 'starbucks', 'subway', 'wings',
                         'mcdonalds', 'burger', 'pizza', 'taco', 'comida', 'food', 'kitchen',
                         'bistro', 'grill', 'diner', 'cantina', 'salad', 'chicken'],
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
            'Tecnología': ['best buy', 'office', 'depot', 'steren', 'radioshack',
                          'apple', 'samsung', 'dell', 'hp'],
            'Otros': []
        }
        
        # Patrones mejorados y más flexibles
        self.patterns = {
            'amount': [
                # Buscar "total" seguido de números con formato de dinero
                r'total\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'total\s+a\s+pagar\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'importe\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'pagar\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                r'neto\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                # Buscar $ seguido de números (común en tickets)
                r'\$\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)',
                # Números con formato de dinero solos
                r'\b(\d{1,3}(?:[,\.]\d{3})*\.\d{2})\b',
            ],
            'date': [
                # Múltiples formatos de fecha
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{4})',
                r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
                r'(\d{1,2}[/-]\d{1,2}[/-]\d{2})',
                r'fecha\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                # Formatos con mes en texto
                r'(\d{1,2}\s+(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)[a-z]*\s+\d{2,4})',
                r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',
            ],
            'time': [
                # Formatos de hora más flexibles
                r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?)',
                r'hora\s*:?\s*(\d{1,2}:\d{2})',
                r'time\s*:?\s*(\d{1,2}:\d{2})',
            ],
            'card': [
                # Patrones más específicos para tarjetas
                r'\*{4}\s*(\d{4})',
                r'xxxx\s*(\d{4})',
                r'card\s*(?:ending)?\s*(?:in)?\s*:?\s*\**(\d{4})',
                r'tarjeta\s*:?\s*\*+(\d{4})',
                r'tc\s*:?\s*\*+(\d{4})',
                r'visa\s*:?\s*(\d{4})',
                r'mastercard\s*:?\s*(\d{4})',
                # Últimos 4 dígitos cerca de palabras clave
                r'(?:terminacion|termina|ending)\s*:?\s*(\d{4})',
            ]
        }
    
    def preprocess_image_multiple_methods(self, image_path):
        """Aplica múltiples técnicas de preprocesamiento y retorna varias versiones"""
        img = cv2.imread(str(image_path))
        
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_images = []
        
        # Método 1: Escala de grises simple
        processed_images.append(('gray', gray))
        
        # Método 2: Umbral binario
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_images.append(('binary', binary))
        
        # Método 3: Umbral adaptativo
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(('adaptive', adaptive))
        
        # Método 4: Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(('otsu', otsu))
        
        # Método 5: Denoise + contraste
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        processed_images.append(('enhanced', enhanced))
        
        # Método 6: Aumentar tamaño (mejor para texto pequeño)
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
        
        # Diferentes configuraciones de Tesseract
        configs = [
            '--oem 3 --psm 6 -l spa+eng',  # Auto con español e inglés
            '--oem 3 --psm 4 -l spa+eng',  # Columna única
            '--oem 3 --psm 3 -l eng',      # Solo inglés
            '--oem 1 --psm 6 -l spa',      # Solo español
        ]
        
        # Probar diferentes combinaciones
        for method_name, processed_img in processed_images[:3]:  # Usar solo las 3 mejores
            for config in configs[:2]:  # Usar solo 2 configs por método
                try:
                    text = pytesseract.image_to_string(processed_img, config=config)
                    if text and len(text.strip()) > 20:  # Solo si hay contenido
                        all_texts.append(text.lower())
                except:
                    continue
        
        # Combinar todos los textos para maximizar chances de encontrar datos
        combined_text = '\n'.join(all_texts)
        return combined_text
    
    def extract_amount(self, text):
        """Extrae el monto con lógica mejorada"""
        amounts = []
        
        for pattern in self.patterns['amount']:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                # Limpiar el monto
                clean_amount = match.replace(',', '').replace(' ', '')
                try:
                    amount = float(clean_amount)
                    # Filtrar montos sospechosos
                    if 0.01 <= amount <= 100000:  # Rango razonable
                        amounts.append(amount)
                except ValueError:
                    continue
        
        if not amounts:
            return None
        
        # Estrategia: buscar "total" explícito primero
        total_pattern = r'total\s*:?\s*\$?\s*(\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?)'
        total_matches = re.findall(total_pattern, text, re.IGNORECASE)
        if total_matches:
            try:
                return float(total_matches[-1].replace(',', ''))
            except:
                pass
        
        # Si no hay "total", usar el monto más grande (suele ser el total)
        return max(amounts)
    
    def extract_date(self, text):
        """Extrae la fecha con mejor manejo de formatos"""
        for pattern in self.patterns['date']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.strip()
                
                # Intentar múltiples formatos
                formats = [
                    '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d',
                    '%d/%m/%y', '%d-%m-%y', '%m/%d/%Y', '%m-%d-%Y',
                    '%d %b %Y', '%d %B %Y', '%d %b %y'
                ]
                
                for fmt in formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        # Validar que la fecha sea razonable
                        if 2020 <= date_obj.year <= 2026:
                            return date_obj.strftime('%Y-%m-%d')
                    except (ValueError, AttributeError):
                        continue
        
        return None
    
    def extract_time(self, text):
        """Extrae la hora con normalización"""
        for pattern in self.patterns['time']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_str = match.group(1).strip()
                # Normalizar formato
                time_str = time_str.replace('.', '').upper()
                return time_str
        return None
    
    def extract_card_last4(self, text):
        """Extrae últimos 4 dígitos con mejor detección"""
        # Primero buscar efectivo
        if re.search(r'\befectivo\b|\bcash\b|\bcontado\b', text, re.IGNORECASE):
            return 'Efectivo'
        
        # Buscar patrones de tarjeta
        for pattern in self.patterns['card']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Retornar el primero que encontremos
                digits = matches[0] if isinstance(matches[0], str) else matches[0]
                if digits.isdigit() and len(digits) == 4:
                    return f"****{digits}"
        
        # Buscar tipos de tarjeta
        card_types = ['visa', 'mastercard', 'amex', 'american express']
        for card_type in card_types:
            if card_type in text.lower():
                return card_type.title()
        
        return None
    
    def extract_merchant(self, text):
        """Extrae el nombre del comercio con mejor heurística"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Buscar en las primeras 10 líneas
        candidates = []
        for i, line in enumerate(lines[:10]):
            # Filtrar líneas que parecen ser nombre de comercio
            if (3 <= len(line) <= 60 and 
                not re.match(r'^[\d\s:\-/\.\$]+$', line) and
                not re.search(r'\d{4,}', line)):  # No tiene muchos números
                
                # Dar más peso a las primeras líneas
                score = 10 - i
                candidates.append((line, score))
        
        if candidates:
            # Ordenar por score y retornar el mejor
            candidates.sort(key=lambda x: x[1], reverse=True)
            merchant = candidates[0][0]
            # Limpiar caracteres extraños
            merchant = re.sub(r'[^\w\s\-\.]', '', merchant)
            return merchant.title()[:50]  # Máximo 50 chars
        
        return 'Desconocido'
    
    def categorize_expense(self, text, merchant):
        """Categoriza con scoring mejorado"""
        text_combined = f"{text} {merchant}".lower()
        
        # Contar matches por categoría
        category_scores = {}
        for category, keywords in self.categories.items():
            score = sum(1 for keyword in keywords if keyword in text_combined)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            # Retornar la categoría con más matches
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return 'Otros'
    
    def analyze_receipt(self, image_path):
        """Analiza un ticket con todas las mejoras"""
        print(f"Procesando: {image_path.name}")
        
        text = self.extract_text_multiple_configs(image_path)
        
        if not text or len(text.strip()) < 20:
            print(f"  ⚠️  Advertencia: Texto extraído muy corto o vacío")
            return {
                'archivo': image_path.name,
                'fecha': None,
                'hora': None,
                'comercio': 'Error OCR',
                'monto': None,
                'metodo_pago': None,
                'categoria': 'Error',
                'texto_extraido': text[:300]
            }
        
        merchant = self.extract_merchant(text)
        amount = self.extract_amount(text)
        date = self.extract_date(text)
        time = self.extract_time(text)
        payment = self.extract_card_last4(text)
        category = self.categorize_expense(text, merchant)
        
        # Logging para debug
        print(f"  ✓ Comercio: {merchant}")
        print(f"  ✓ Monto: ${amount if amount else 'N/A'}")
        print(f"  ✓ Fecha: {date if date else 'N/A'}")
        print(f"  ✓ Categoría: {category}")
        
        return {
            'archivo': image_path.name,
            'fecha': date,
            'hora': time,
            'comercio': merchant,
            'monto': amount,
            'metodo_pago': payment,
            'categoria': category,
            'texto_extraido': text[:300]  # Primeros 300 chars
        }
    
    def process_folder(self, folder_path):
        """Procesa todos los tickets en una carpeta"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"Error: La carpeta {folder_path} no existe")
            return None
        
        # Buscar todas las imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        images = []
        for ext in image_extensions:
            images.extend(folder.glob(f'*{ext}'))
            images.extend(folder.glob(f'**/*{ext}'))  # Buscar en subcarpetas también
        
        if not images:
            print(f"No se encontraron imágenes en {folder_path}")
            return None
        
        print(f"\n{'='*60}")
        print(f"Encontradas {len(images)} imágenes")
        print(f"{'='*60}\n")
        
        expenses = []
        for img_path in images:
            expense = self.analyze_receipt(img_path)
            expenses.append(expense)
            print()  # Línea en blanco entre tickets
        
        return pd.DataFrame(expenses)
    
    def generate_report(self, df, output_file='expenses_report.csv'):
        """Genera reporte mejorado con más detalles"""
        if df is None or df.empty:
            print("No hay datos para generar reporte")
            return
        
        # Guardar CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n{'='*60}")
        print(f"✓ Reporte guardado en: {output_file}")
        print(f"{'='*60}")
        
        # Estadísticas detalladas
        print("\n" + "="*60)
        print("📊 RESUMEN DE GASTOS")
        print("="*60)
        
        # Total
        valid_amounts = df[df['monto'].notna()]
        total = valid_amounts['monto'].sum()
        print(f"\n💰 Total de gastos: ${total:,.2f}")
        print(f"📄 Tickets procesados: {len(df)}")
        print(f"✓ Tickets con monto: {len(valid_amounts)} ({len(valid_amounts)/len(df)*100:.1f}%)")
        print(f"✓ Tickets con fecha: {df['fecha'].notna().sum()} ({df['fecha'].notna().sum()/len(df)*100:.1f}%)")
        print(f"✓ Tickets con método de pago: {df['metodo_pago'].notna().sum()} ({df['metodo_pago'].notna().sum()/len(df)*100:.1f}%)")
        
        # Por categoría
        if len(valid_amounts) > 0:
            print("\n" + "-"*60)
            print("📂 GASTOS POR CATEGORÍA")
            print("-"*60)
            by_category = valid_amounts.groupby('categoria')['monto'].agg(['sum', 'count', 'mean'])
            by_category = by_category.sort_values('sum', ascending=False)
            for cat, row in by_category.iterrows():
                percentage = (row['sum'] / total) * 100
                print(f"{cat:20s}: ${row['sum']:8,.2f} ({percentage:5.1f}%) - {int(row['count'])} tickets - Promedio: ${row['mean']:,.2f}")
            
            # Top comercios
            print("\n" + "-"*60)
            print("🏪 TOP 5 COMERCIOS")
            print("-"*60)
            by_merchant = valid_amounts.groupby('comercio')['monto'].agg(['sum', 'count'])
            by_merchant = by_merchant.sort_values('sum', ascending=False).head(5)
            for i, (merchant, row) in enumerate(by_merchant.iterrows(), 1):
                print(f"{i}. {merchant:30s}: ${row['sum']:8,.2f} ({int(row['count'])} tickets)")
            
            # Estadísticas
            print("\n" + "-"*60)
            print("📈 ESTADÍSTICAS")
            print("-"*60)
            avg = valid_amounts['monto'].mean()
            median = valid_amounts['monto'].median()
            max_expense = valid_amounts['monto'].max()
            min_expense = valid_amounts['monto'].min()
            print(f"Promedio por ticket: ${avg:,.2f}")
            print(f"Mediana: ${median:,.2f}")
            print(f"Gasto máximo: ${max_expense:,.2f}")
            print(f"Gasto mínimo: ${min_expense:,.2f}")
        
        return df


def main():
    """Función principal"""
    import sys
    
    print("\n" + "="*60)
    print("💸 EXPENSE TRACKER - Análisis de Tickets con OCR")
    print("="*60)
    
    tracker = ExpenseTracker()
    
    # Carpeta de tickets
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("\nIngresa la ruta de la carpeta con tickets (o Enter para './tickets'): ").strip()
        if not folder_path:
            folder_path = './tickets'
    
    # Procesar tickets
    df = tracker.process_folder(folder_path)
    
    # Generar reporte
    if df is not None:
        output_file = 'expenses_report.csv'
        tracker.generate_report(df, output_file)
        
        print("\n" + "="*60)
        print("✅ PROCESO COMPLETADO")
        print("="*60)
        print(f"\n📁 Archivo generado: {output_file}")
        print("💡 Tip: Abre el CSV en Excel o Google Sheets para análisis detallado")
        print()


if __name__ == "__main__":
    main()