import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import requests
from datetime import datetime, timedelta


class Prediction():
    def __init__(self):
        print("Initialisation du modèle")
        self.modele = joblib.load('../models/gradientboosting.pkl')

    def get_GDD_from_openmeteo(self, lat, lon, date, t_base=10):
        """
        Récupère les températures et calcule le GDD cumulé depuis le 1er janvier
        ainsi que le chilling hivernal cumulé depuis le 1er novembre N-1.
        Retourne un DataFrame complet avec une ligne par jour.
        """
        target = pd.to_datetime(date)
        year = target.year
        chilling_start = f"{year - 1}-11-01"
        gdd_start = f"{year}-01-01"

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': chilling_start,
            'end_date': date,
            'daily': 'temperature_2m_max,temperature_2m_min',
            'timezone': 'auto'
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        temps = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            't_max': data['daily']['temperature_2m_max'],
            't_min': data['daily']['temperature_2m_min']
        })

        temps['t_mean'] = (temps['t_max'] + temps['t_min']) / 2

        temps['gdd_daily'] = temps['t_mean'].where(temps['date'] >= gdd_start, other=0)
        temps['gdd_daily'] = (temps['gdd_daily'] - t_base).clip(lower=0)
        temps['gdd_cumul'] = temps['gdd_daily'].cumsum()

        temps['chilling_daily'] = temps['t_mean'].apply(lambda x: 1 if 0 < x <= 7.2 else 0)
        temps['chilling_hivernal'] = temps['chilling_daily'].cumsum()

        return temps

    def build_prediction(self, lat, lon, alt, year):
        """
        Calcule l'évolution des prédictions sur toute l'année (365 itérations).
        Retourne la date moyenne prédite et un graphique en base64.
        """
        year_end = min(f"{year}-12-31", datetime.today().strftime("%Y-%m-%d"))
        pd_meteo = self.get_GDD_from_openmeteo(lat, lon, year_end)
        pd_meteo_year = pd_meteo[pd_meteo['date'] >= f"{year}-01-01"].reset_index(drop=True)

        jours = []
        predictions = []
        predictions_avg20 = []

        for i in range(365):
            if i < len(pd_meteo_year):
                row_meteo = pd_meteo_year.iloc[i]
            else:
                row_meteo = pd_meteo_year.iloc[-1]

            pds = pd.DataFrame([{
                "latitude": lat,
                "longitude": lon,
                "altitude": alt,
                "jour_n": i,
                "temps_thermique10": row_meteo["gdd_cumul"],
                "chilling_hivernal": row_meteo["chilling_hivernal"],
                "annee": year
            }])
            delta = self.modele.predict(pds)[0]
            pred = i + delta

            jours.append(i)
            predictions.append(pred)
            last_20 = predictions[-20:]
            predictions_avg20.append(sum(last_20) / len(last_20))

        jour_prevu = int(round(np.mean(predictions)))
        date_prevue = datetime(year, 1, 1) + timedelta(days=jour_prevu - 1)

        graph_b64 = self._generate_graph(lat, lon, alt, jours, predictions, predictions_avg20, jour_prevu)

        return {
            'date_moyenne': date_prevue.strftime('%Y-%m-%d'),
            'jour_prevu': jour_prevu,
            'graph': graph_b64
        }

    def _generate_graph(self, lat, lon, alt, jours, predictions, predictions_avg20, jour_prevu):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e2330')
        ax.set_facecolor('#141923')

        ax.plot(jours, predictions, linewidth=1, color='#4a90e2', alpha=0.5, label='Prévision journalière')
        ax.plot(jours, predictions_avg20, linestyle='--', linewidth=2, color='#f59e0b', label='Moyenne glissante (20 j)')
        ax.axhline(y=jour_prevu, color='#10b981', linestyle='--', linewidth=1.5, label=f"Moyenne prévue (J{jour_prevu})")

        y_center = jour_prevu
        ax.set_ylim(max(0, y_center - 60), min(365, y_center + 60))

        ax.set_xlabel("Jour de l'année", color='#b0b8c4')
        ax.set_ylabel("Jour de l'année prévu", color='#b0b8c4')
        ax.set_title(f"Évolution de la prévision — lat: {lat:.4f}  lon: {lon:.4f}  alt: {alt}m", color='#e8eaed')
        ax.tick_params(colors='#8b95a5')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3748')
        ax.legend(facecolor='#1e2330', labelcolor='#e8eaed', edgecolor='#2d3748')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor())
        buf.seek(0)
        graph_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return graph_b64

    def makePrediction(self, year, latitude, longitude, altitude):
        return self.build_prediction(latitude, longitude, altitude, year)
