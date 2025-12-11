import pytesseract

try:
    # Intenta obtener una lista de idiomas disponibles
    languages = pytesseract.get_languages(config='')
    print("Idiomas detectados por Tesseract:")
    print(languages)

    if 'spa' in languages and 'eng' in languages:
        print("\n✅ ¡Confirmado! Los datos de 'spa' y 'eng' están disponibles.")
    else:
        print("\n❌ Advertencia: Algunos idiomas podrían faltar. Revisa la carpeta tessdata.")

except pytesseract.TesseractNotFoundError:
    print("❌ Error: Tesseract no está instalado o no está en el PATH de tu sistema.")
    print("Por favor, asegúrate de que el ejecutable de Tesseract esté instalado.")