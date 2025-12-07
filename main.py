from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import re
import matplotlib
matplotlib.use('Agg') # <-- КРИТИЧЕСКАЯ СТРОКА!
import matplotlib.pyplot as plt
import io
import base64

# --- Настройка FastAPI ---
app = FastAPI()

class PumpData(BaseModel):
    raw_text: str

@app.get("/")
def home():
    return {"status": "RusPump API (Cubic Poly + Graph) is running"}

@app.post("/calculate_poly")
def calculate_poly(data: PumpData):
    try:
        # 1. Парсинг
        numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', data.raw_text)]

        if len(numbers) < 8 or len(numbers) % 2 != 0:
            return {"error": "Ошибка: Для построения кубического полинома требуется минимум 4 пары чисел (Q H)."}

        q_user = np.array(numbers[0::2])
        h_user = np.array(numbers[1::2])

        # 2. Математика (Полином 3-й степени)
        coefficients = np.polyfit(q_user, h_user, 3)
        poly_func = np.poly1d(coefficients)
        a, b, c, d = coefficients

        # Расчет R^2
        h_pred = poly_func(q_user)
        y_mean = np.mean(h_user)
        ss_tot = np.sum((h_user - y_mean)**2)
        ss_res = np.sum((h_user - h_pred)**2)
        r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

        # 3. ГЕНЕРАЦИЯ ГРАФИКА
        plt.figure(figsize=(10, 6))

        # Строим плавную кривую (Q max + 10%)
        q_smooth = np.linspace(0, max(q_user) * 1.1, 100)
        h_smooth = poly_func(q_smooth)

        # Визуализация (Стандарт RusPump)
        plt.plot(q_smooth, h_smooth, 'b-', label='Аппроксимация (Полином 3 ст.)', linewidth=2)
        plt.plot(q_user, h_user, 'ro', label='Исходные точки', markersize=8)

        plt.title(f"Характеристика насоса Q-H (R²={r_squared:.4f})")
        plt.xlabel("Расход Q, м³/ч")
        plt.ylabel("Напор H, м")
        plt.grid(True, which='both', linestyle='--')
        plt.legend()

        # Сохранение в буфер памяти
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Кодируем в Base64 для передачи через JSON
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # 4. Формирование текста ответа
        quality_emoji = "✅" if r_squared > 0.98 else "⚠️" if r_squared > 0.9 else "❌"
        sb = "+" if b >= 0 else ""
        sc = "+" if c >= 0 else ""
        sd = "+" if d >= 0 else ""
        visual_eq = f"H(Q) = {a:.5f}Q³ {sb}{b:.5f}Q² {sc}{c:.5f}Q {sd}{d:.2f}"

        ex_dot = f"=({a:.10f})*A1^3 + ({b:.10f})*A1^2 + ({c:.10f})*A1 + ({d:.10f})"
        ex_comma = ex_dot.replace(".", ",")

        return {
            "image_base64": image_base64,
            "message": (
                f"📈 *Результат расчета:*\n`{visual_eq}`\n"
                f"{quality_emoji} **Точность (R²):** {r_squared*100:.2f}%\n\n"
                f"📋 *Excel (EN):* `{ex_dot}`\n"
                f"🇷🇺 *Excel (RU):* `{ex_comma}`"
            )
        }

    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) # Используем порт 5000, который вы зафиксировали