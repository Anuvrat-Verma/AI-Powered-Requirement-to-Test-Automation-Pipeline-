import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import os


class SeleniumComplianceEvaluator:
    def __init__(self):
        self.model_path = "Backend/models/compliance_model.keras"
        self.model = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        if os.path.exists(self.model_path):
            print("✅ Loading saved compliance model...")
            self.model = tf.keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
        else:
            print("🔄 Training compliance model with LSTM for Amazon Prime Day (first time only)...")
            self._train_model()
            self.model.save(self.model_path)
            print(f"✅ Model saved at {self.model_path}!")

    def _train_model(self):
        # Training data tailored for Amazon Prime Day Deals workflows
        scripts = [
            # === GOOD / COMPLIANT (Amazon Prime Day style) ===
            "class AmazonPrimeDayDealsPage:\n    def __init__(self, driver):\n        self.driver = driver\n    def navigate_to_amazon_prime_day_deals(self):\n        self.driver.get('https://example.com/amazon-prime-day-deals')",
            "def filter_deals_by_category(self, category):\n    category_filter_locator = (By.CSS_SELECTOR, f\"div.category-filter[aria-label='{category}']\")\n    WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable(category_filter_locator)).click()",
            "search_input_locator = (By.ID, 'search-input')\nWebDriverWait(self.driver, 10).until(EC.presence_of_element_located(search_input_locator)).send_keys('Smart TV')",
            "button_locator = (By.XPATH, \"//button[@aria-label='Shop Now or Learn More']\")\nWebDriverWait(self.driver, 10).until(EC.element_to_be_clickable(button_locator)).click()",
            "assert WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.ID, 'amazon-prime-day-deals-container')))",
            "assert WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div.deal-item.electronics')))",
            "assert WebDriverWait(self.driver, 10).until(EC.url_contains('product-details-page'))",

            # More good POM + explicit wait patterns
            "class AmazonPrimeDayDealsPage:\n    def __init__(self, driver):\n        self.driver = driver\n    def click_shop_now(self):\n        locator = (By.CSS_SELECTOR, 'button.shop-now')\n        WebDriverWait(self.driver, 15).until(EC.element_to_be_clickable(locator)).click()",

            # === BAD / NON-COMPLIANT ===
            "def test_login(driver):\n    driver.find_element(By.ID, 'user').send_keys('admin')\n    time.sleep(5)\n    assert True",
            "from selenium import webdriver\ndriver = webdriver.Chrome()\ndriver.get('url')\nassert 'Login' in driver.title",
            "driver.find_element(By.XPATH, '//div/p/a/button').click()\ntime.sleep(10)",
            "def test_click(driver):\n    driver.find_element(By.XPATH, '//button').click()\n    assert True",
            "time.sleep(20)\ndriver.find_element(By.ID, 'submit').click()",
            "driver.get('https://example.com/amazon-prime-day-deals')\nassert driver.title == 'Amazon'",
            "element = driver.find_element(By.XPATH, '//*')\nelement.click()",
            "time.sleep(15)\nassert True"
        ]

        # Labels: 8 Good, 8 Bad
        labels = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        # Vectorizer
        vectorizer = layers.TextVectorization(
            max_tokens=3000,
            output_sequence_length=250,
            output_mode='int'
        )
        vectorizer.adapt(np.array(scripts))

        # Model with LSTM
        self.model = models.Sequential([
            layers.Input(shape=(), dtype=tf.string),
            vectorizer,
            layers.Embedding(3000, 64, mask_zero=True),

            # LSTM for better understanding of Prime Day workflows
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),

            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')]
        )

        self.model.fit(np.array(scripts, dtype=object), labels, epochs=150, verbose=0)
        print("✅ Amazon Prime Day Compliance Classifier Trained with LSTM!")

    def evaluate(self, generated_code: str):
        if not generated_code or len(generated_code.strip()) < 20:
            return "⚠️ Code too short to evaluate"

        input_data = np.array([generated_code], dtype=object)

        prediction = self.model.predict(input_data, verbose=0)
        score = float(prediction[0][0])

        if score > 0.75:
            return f"✅ COMPLIANT (Score: {score:.3f})"
        else:
            return f"❌ REJECTED: Violates Selenium Standards (Score: {score:.3f})"


# Initialize
selenium_evaluator = SeleniumComplianceEvaluator()

# Test
result = selenium_evaluator.evaluate("def test(): time.sleep(10)")
print(result)