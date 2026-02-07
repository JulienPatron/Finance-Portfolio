import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIGURATION ---
HOME_URL = "https://julien-patron-projects.streamlit.app"
PROJECT_URL = "https://julien-patron-projects.streamlit.app/F1_Elo_System"

def wake_up_app():
    # 1. JITTER : Délai aléatoire (30s à 3min) pour simuler un humain
    delay = random.randint(60, 180)
    print(f"🕒 Pause aléatoire de {delay} secondes...")
    time.sleep(delay)

    print("🚀 Démarrage du robot...")
    
    # Options Chrome Headless
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"🌍 Connexion à : {HOME_URL}")
        driver.get(HOME_URL)
        
        # On attend un peu que le DOM se construise
        time.sleep(5)
        
        # --- STRATÉGIE CIBLÉE (data-testid) ---
        print("🔎 Recherche du bouton de réveil via data-testid...")
        
        try:
            # On cherche spécifiquement l'attribut que tu m'as donné
            selector = '[data-testid="wakeup-button-viewer"]'
            buttons = driver.find_elements(By.CSS_SELECTOR, selector)
            
            if buttons:
                print("💤 DÉTECTION : L'application est en veille.")
                print(f"🎯 Bouton trouvé via '{selector}'")
                
                # Clic
                buttons[0].click()
                
                print("👆 CLIC EFFECTUÉ. Attente du redémarrage...")
                time.sleep(15)
                
                # Vérification post-clic (optionnel)
                print("✅ Mission accomplie.")
                return
            else:
                print("⚡ Pas de bouton de veille détecté (l'app est éveillée).")

        except Exception as e:
            print(f"Info: Erreur lors de la recherche du bouton ({e})")

        # --- PHASE B : Maintenance d'activité ---
        print("🔄 Navigation vers un projet pour maintenir l'activité...")
        try:
            driver.get(PROJECT_URL)
            time.sleep(8)
            print("✅ Visite terminée.")
        except Exception as e:
            print(f"❌ Erreur navigation : {e}")

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : {e}")
    finally:
        driver.quit()
        print("🏁 Fin du script.")

if __name__ == "__main__":
    wake_up_app()