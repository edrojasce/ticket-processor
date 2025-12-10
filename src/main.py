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
            'Alimentos': ['oxxo', 'seven', '7-eleven', 'walmart', 'soriana', 'chedraui', 
                         'superama', 'bodega', 'aurrera', 'costco', 'sams', 'heb',
                         'restaurant', 'cafe', 'coffee', 'starbucks', 'subway',
                         'mcdonalds', 'burger', 'pizza', 'taco', 'comida'],
            'Transporte': ['uber', 'didi', 'gas', 'gasolina', 'pemex', 'shell', 
                          'mobil', 'bp', 'taxi', 'parking', 'estacionamiento'],
            'Entretenimiento': ['cinema', 'cine', 'cinepolis', 'cinemex', 'netflix',
                               'spotify', 'xbox', 'playstation', 'steam'],
            'Salud': ['farmacia', 'pharmacy', 'guadalajara', 'ahorro', 'similares',
                     'benavides', 'hospital', 'doctor', 'clinic', 'medic'],
            'Servicios': ['electric', 'cfe', 'agua', 'telmex', 'telcel', 'att',
                         'movistar', 'izzi', 'totalplay', 'dish'],
            'Ropa': ['zara', 'h&m', 'liverpool', 'palacio', 'coppel', 'suburbia',
                    'nike', 'adidas', 'clothing', 'shoes'],
            'Tecnología': ['best buy', 'office', 'depot', 'steren', 'radioshack',
                          'apple', 'samsung', 'dell', 'hp'],
            'Otros': []
        }
        
        # Patrones comunes en tickets mexicanos y de otros países
        self.patterns = {
            'amount': [
                r'total[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'total[:\s]*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'importe[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'pagar[:\s]*\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            ],
            'date': [
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                r'fecha[:\s]*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
            ],
            'time': [
                r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?)',
                r'hora[:\s]*(\d{1,2}:\d{2})',
            ],
            'card': [
                r'(?:tarjeta|card|tc).*?(\d{4})',
                r'\*+(\d{4})',
                r'xxxx.*?(\d{4})',
            ]
        }
    
    def preprocess_image(self, image_path):
        """Preprocesa la imagen para mejorar la calidad del OCR"""
        img = cv2.imread(str(image_path))
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbral adaptativo
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Reducir ruido
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Aumentar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def extract_text_from_image(self, image_path):
        """Extrae texto de la imagen usando Tesseract OCR"""
        try:
            # Preprocesar imagen
            processed_img = self.preprocess_image(image_path)
            
            # Configurar Tesseract para español e inglés
            custom_config = r'--oem 3 --psm 6 -l spa+eng'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            return text.lower()
        except Exception as e:
            print(f"Error al procesar {image_path}: {e}")
            return ""
    
    def extract_amount(self, text):
        """Extrae el monto del ticket"""
        for pattern in self.patterns['amount']:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Tomar el último monto (usualmente es el total)
                amount_str = matches[-1].replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    continue
        return None
    
    def extract_date(self, text):
        """Extrae la fecha del ticket"""
        for pattern in self.patterns['date']:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                # Intentar varios formatos de fecha
                for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d',
                           '%d/%m/%y', '%d-%m-%y']:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        return date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
        return None
    
    def extract_time(self, text):
        """Extrae la hora del ticket"""
        for pattern in self.patterns['time']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def extract_card_last4(self, text):
        """Extrae los últimos 4 dígitos de la tarjeta"""
        for pattern in self.patterns['card']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Buscar "efectivo" o "cash"
        if re.search(r'efectivo|cash', text, re.IGNORECASE):
            return 'Efectivo'
        
        return None
    
    def extract_merchant(self, text):
        """Intenta identificar el comercio"""
        lines = text.split('\n')
        # Las primeras líneas suelen tener el nombre del comercio
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 3 and len(line) < 50:
                # Verificar si no es fecha, hora o número
                if not re.match(r'^[\d\s:\-/]+$', line):
                    return line.title()
        return 'Desconocido'
    
    def categorize_expense(self, text, merchant):
        """Categoriza el gasto basado en palabras clave"""
        text_combined = f"{text} {merchant}".lower()
        
        for category, keywords in self.categories.items():
            for keyword in keywords:
                if keyword in text_combined:
                    return category
        
        return 'Otros'
    
    def analyze_receipt(self, image_path):
        """Analiza un ticket completo"""
        print(f"Procesando: {image_path.name}")
        
        text = self.extract_text_from_image(image_path)
        
        if not text:
            return {
                'archivo': image_path.name,
                'fecha': None,
                'hora': None,
                'comercio': 'Error',
                'monto': None,
                'metodo_pago': None,
                'categoria': 'Error',
                'texto_extraido': ''
            }
        
        merchant = self.extract_merchant(text)
        
        return {
            'archivo': image_path.name,
            'fecha': self.extract_date(text),
            'hora': self.extract_time(text),
            'comercio': merchant,
            'monto': self.extract_amount(text),
            'metodo_pago': self.extract_card_last4(text),
            'categoria': self.categorize_expense(text, merchant),
            'texto_extraido': text[:200]  # Primeros 200 chars para debug
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
        
        if not images:
            print(f"No se encontraron imágenes en {folder_path}")
            return None
        
        print(f"\nEncontradas {len(images)} imágenes")
        print("="*60)
        
        expenses = []
        for img_path in images:
            expense = self.analyze_receipt(img_path)
            expenses.append(expense)
        
        return pd.DataFrame(expenses)
    
    def generate_report(self, df, output_file='expenses_report.csv'):
        """Genera reporte de gastos"""
        if df is None or df.empty:
            print("No hay datos para generar reporte")
            return
        
        # Guardar CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✓ Reporte guardado en: {output_file}")
        
        # Estadísticas
        print("\n" + "="*60)
        print("RESUMEN DE GASTOS")
        print("="*60)
        
        # Total
        valid_amounts = df[df['monto'].notna()]
        total = valid_amounts['monto'].sum()
        print(f"\nTotal de gastos: ${total:,.2f}")
        print(f"Tickets procesados: {len(df)}")
        print(f"Tickets con monto: {len(valid_amounts)}")
        
        # Por categoría
        print("\n--- Gastos por Categoría ---")
        by_category = valid_amounts.groupby('categoria')['monto'].agg(['sum', 'count'])
        by_category = by_category.sort_values('sum', ascending=False)
        for cat, row in by_category.iterrows():
            print(f"{cat:20s}: ${row['sum']:8,.2f} ({int(row['count'])} tickets)")
        
        # Por comercio
        print("\n--- Top 5 Comercios ---")
        by_merchant = valid_amounts.groupby('comercio')['monto'].agg(['sum', 'count'])
        by_merchant = by_merchant.sort_values('sum', ascending=False).head(5)
        for merchant, row in by_merchant.iterrows():
            print(f"{merchant:20s}: ${row['sum']:8,.2f}")
        
        # Promedio
        avg = valid_amounts['monto'].mean()
        print(f"\nPromedio por ticket: ${avg:,.2f}")
        
        return df


def main():
    """Función principal"""
    import sys
    
    tracker = ExpenseTracker()
    
    # Carpeta de tickets (puedes cambiar esto)
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Ingresa la ruta de la carpeta con tickets (o Enter para './tickets'): ").strip()
        if not folder_path:
            folder_path = './tickets'
    
    print(f"\nAnalizando tickets en: {folder_path}")
    print("="*60)
    
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
        print("\nPuedes abrir el CSV en Excel o Google Sheets para análisis detallado")


if __name__ == "__main__":
    main()