# Tema A - descarga el registro oficial SUNASS de interrupciones de agua/alcantarillado.
# datosabiertos.gob.pe bloquea requests normales con un WAF (devuelve 418), por eso se
# usa un navegador real via Playwright para que la descarga pase el challenge.

from playwright.sync_api import sync_playwright

URL_DATASET = "https://www.datosabiertos.gob.pe/dataset/registro-de-interrupciones-del-servicio-de-agua-y-alcantarillado-imprevistas-y-programadas"
DEST = "data/tema_a/Interrupciones_Dataset.csv"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(URL_DATASET, timeout=60000)
    page.wait_for_load_state("networkidle")

    with page.expect_download() as download_info:
        page.click("text=Descargar")
    download = download_info.value
    download.save_as(DEST)

    browser.close()

print(f"guardado en {DEST}")
