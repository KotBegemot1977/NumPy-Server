from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn
import re

app = FastAPI()

class PumpData(BaseModel):
    raw_text: str

@app.get("/")
def home():
    return {"status": "RusPump API (Stable Text Only) is running"}

@app.post("/calculate_poly")
def calculate_poly(data: PumpData):
    try:
        # 1. Парсинг
        numbers = [float(s) for s in re.findall(r'-?\d+\.?\d*', data.raw_text)]

        if len(numbers) < 8 or len(numbers) % 2 != 0:
            return {"error": "Нужно минимум 4 точки (8 чисел)."}

        q_vals = np.array(numbers[0::2])
        h_vals = np.array(numbers[1::2])

        # 2. Расчет полинома 3-й степени
        coefficients = np.polyfit(q_vals, h_vals, 3)
        a, b, c, d = coefficients

        # 3. Расчет R^2
        poly_func = np.poly1d(coefficients)
        h_pred = poly_func(q_vals)
        y_mean = np.mean(h_vals)
        ss_tot = np.sum((h_vals - y_mean)**2)
        ss_res = np.sum((h_vals - h_pred)**2)

        if ss_tot == 0:
            r_squared = 1.0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        quality_emoji = "✅" if r_squared > 0.98 else "⚠️" if r_squared > 0.9 else "❌"

        # 4. Формирование формул
        sb = "+" if b >= 0 else ""
        sc = "+" if c >= 0 else ""
        sd = "+" if d >= 0 else ""
        visual_eq = f"H(Q) = {a:.5f}Q³ {sb}{b:.5f}Q² {sc}{c:.5f}Q {sd}{d:.2f}"

        ex_dot = f"=({a:.10f})*A1^3 + ({b:.10f})*A1^2 + ({c:.10f})*A1 + ({d:.10f})"
        ex_comma = ex_dot.replace(".", ",")

        return {
            "r2": r_squared,
            "message": (
                f"📈 *Результат расчета:*\n\n"
                f"`{visual_eq}`\n\n"
                f"{quality_emoji} **R²:** {r_squared*100:.2f}%\n\n"
                f"📋 *Для Excel (EN/IT - точка):*\n"
                f"`{ex_dot}`\n\n"
                f"🇷🇺 *Для Excel (RU - запятая):*\n"
                f"`{ex_comma}`\n\n"
                f"_(В формулах Q заменено на ячейку A1)_"
            )
        }

    except Exception as e:
        return {"error": f"Ошибка: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)