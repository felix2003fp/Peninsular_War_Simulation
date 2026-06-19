"""
enrich_map.py
Reads nodes.csv and edges.csv from the same folder, then writes enriched versions
with terrain, node classification, strategic importance, distances, and road type.

Run:  python enrich_map.py
Output: nodes.csv and edges.csv are overwritten in place.
"""

import csv
import math
from pathlib import Path

HERE = Path(__file__).parent   # same folder as this script

# ── Haversine ────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 1)

# ── Node attributes ───────────────────────────────────────────────────────────
# (terra1, terra2, node_type, strategic_importance)
#
# terra1 :  F=Flat  R=Rolling  G=Rugged  O=Other/NA
# terra2 :  B=Bare  M=Mixed    W=Heavily Wooded  D=Desert  O=Other/NA
# node_type: capital | major_city | city | town | intersection
# strategic_importance: 5 (capital) → 4 (major city) → 3 (city) → 2 (town) → 1 (intersection)
#
# Supply-hub logic (for logistics distance):  node_type in {capital, major_city, city}

NODE_ATTRS = {
    # ── node_type / strategic_importance based on population ~1800 ────────────
    # intersection=1 | town <8k=2 | city 8-19k=3 | major_city 20-49k=4
    # regional_capital ≥50k (not Madrid/Lisbon)=5 | capital (Madrid/Lisbon)=6
    #
    # terra1: F=Flat  R=Rolling  G=Rugged  O=Other
    # terra2: B=Bare  M=Mixed    W=Heavily Wooded  D=Desert  O=Other

    # ── CATALONIA ──────────────────────────────────────────────────────────
    'URG': ('R', 'M', 'town', 2),  # Urgell, ~3-4k
    'SOL': ('R', 'M', 'town', 2),  # Solsona, ~3k
    'BER': ('R', 'W', 'town', 2),  # Berga, 3,259
    'RIP': ('G', 'W', 'town', 2),  # Ripoll, ~2-3k
    'VIC': ('R', 'M', 'city', 3),  # Vic, 8,919
    'GIR': ('R', 'M', 'city', 3),  # Girona, 8,014
    'HOS': ('R', 'M', 'town', 2),  # Hostalrich, small
    'MAN': ('R', 'M', 'city', 3),  # Manresa, 8,135
    'BAR': ('F', 'M', 'regional_capital', 5),  # Barcelona, 92,385
    'MAR': ('R', 'M', 'town', 2),  # Martorell, small
    'VIF': ('R', 'M', 'town', 2),  # Vilafranca del Penedès, 3,673
    'VIN': ('F', 'M', 'town', 2),  # Vilanova, small
    'TAR': ('F', 'M', 'city', 3),  # Tarragona, 8,541
    'TOR': ('F', 'M', 'city', 3),  # Tortosa, 16,144
    'CER': ('R', 'B', 'town', 2),  # Cervera, ~3k
    'LLE': ('F', 'B', 'city', 3),  # Lleida, 10,390
    'IN1': ('F', 'B', 'intersection', 1),  # Intersecció Lleida-Cervera-Tarragona
    'VIE': ('G', 'M', 'town', 2),  # Viella, very small
    'LJQ': ('R', 'M', 'town', 2),  # La Jonquera, small border post

    # ── LEVANTE ────────────────────────────────────────────────────────────
    'CAS': ('F', 'M', 'city', 3),  # Castellón, 12,003
    'CAF': ('R', 'M', 'town', 2),  # Castellfort, very small
    'MUR': ('F', 'M', 'town', 2),  # Murviedo (Sagunto), 5,839
    'VAL': ('F', 'M', 'regional_capital', 5),  # Valencia, 100,657
    'SEG': ('R', 'M', 'town', 2),  # Segorbe, 5,321
    'GAN': ('F', 'M', 'town', 2),  # Gandia, 5,798
    'SAF': ('R', 'M', 'city', 3),  # San Felipe (Xàtiva), 12,655
    'XIX': ('R', 'B', 'town', 2),  # Xixona, ~4,906
    'ALI': ('F', 'M', 'city', 3),  # Alicante, 17,760
    'ORI': ('F', 'B', 'major_city', 4),  # Orihuela, 22,913
    'CHI': ('F', 'B', 'town', 2),  # Chinchilla, 3,906
    'ALB': ('F', 'B', 'town', 2),  # Albacete, 7,885 (just under 8k)
    'YEC': ('R', 'B', 'city', 3),  # Yecla, 8,381
    'JUM': ('R', 'B', 'town', 2),  # Jumilla, 6,577
    'MRC': ('F', 'B', 'regional_capital', 5),  # Murcia, 65,515
    'MUL': ('R', 'B', 'town', 2),  # Mula, 6,491
    'CAR': ('R', 'B', 'city', 3),  # Caravaca, 10,990
    'LOR': ('R', 'B', 'major_city', 4),  # Lorca, 37,834
    'CTG': ('F', 'M', 'major_city', 4),  # Cartagena, 29,714

    # ── ARAGON ─────────────────────────────────────────────────────────────
    'VER': ('G', 'M', 'town', 2),  # Verdun/Canfranc, very small
    'VNS': ('G', 'B', 'town', 2),  # Benasque, very small
    'JAC': ('G', 'M', 'town', 2),  # Jaca, ~3-4k (fortress, small)
    'AIN': ('G', 'M', 'town', 2),  # Ainsa, very small
    'AYE': ('R', 'M', 'town', 2),  # Ayerbe, small
    'HUE': ('R', 'M', 'town', 2),  # Huesca, 6,885
    'BRB': ('R', 'M', 'town', 2),  # Barbastro, 5,318
    'LIC': ('F', 'B', 'town', 2),  # Leciñena, very small
    'ZAR': ('F', 'B', 'major_city', 4),  # Zaragoza, 42,600
    'FRA': ('F', 'B', 'town', 2),  # Fraga, ~5-6k
    'PAL': ('R', 'B', 'town', 2),  # Palomares, small
    'TER': ('R', 'B', 'town', 2),  # Teruel, 6,270
    'ALR': ('G', 'M', 'town', 2),  # Albarracín, very small
    'DAR': ('R', 'B', 'town', 2),  # Daroca, small
    'CAL': ('R', 'B', 'city', 3),  # Calatayud, 8,544
    'BOR': ('R', 'B', 'town', 2),  # Borja, ~4-5k
    'TAZ': ('R', 'B', 'town', 2),  # Tarazona, 6,954

    # ── NAVARRA & RIOJA ────────────────────────────────────────────────────
    'MAY': ('G', 'W', 'town', 2),  # Maya, very small pass village
    'PAM': ('R', 'M', 'city', 3),  # Pamplona, ~12-15k est.
    'SGS': ('R', 'M', 'town', 2),  # Sangüesa, small
    'CAP': ('F', 'B', 'town', 2),  # Caparroso, small
    'TUD': ('F', 'B', 'city', 3),  # Tudela, ~8-9k est. (1850=9,148)
    'EST': ('R', 'M', 'town', 2),  # Estella, ~5k
    'VNA': ('R', 'M', 'town', 2),  # Viana, small
    'CLA': ('F', 'B', 'town', 2),  # Calahorra, 5,002
    'LOG': ('R', 'M', 'town', 2),  # Logroño, 6,303
    'AGR': ('R', 'B', 'town', 2),  # Ágreda, small

    # ── BASQUE COUNTRY ─────────────────────────────────────────────────────
    'FUE': ('R', 'W', 'town', 2),  # Fuenterrabía, ~2-3k
    'TOL': ('R', 'W', 'town', 2),  # Tolosa, 4,396
    'ERM': ('R', 'W', 'town', 2),  # Ermua, very small
    'BIL': ('R', 'W', 'city', 3),  # Bilbao, 11,193
    'ORD': ('G', 'W', 'town', 2),  # Orduña, small
    'SAL': ('R', 'M', 'town', 2),  # Salvatierra, small
    'VIT': ('R', 'M', 'town', 2),  # Vitoria, 6,302 (decisive battle, but small town)
    'MIR': ('R', 'M', 'town', 2),  # Miranda de Ebro, small
    'CAZ': ('R', 'M', 'town', 2),  # Calzada area, small
    'FRI': ('G', 'M', 'town', 2),  # Frías, very small

    # ── CANTABRIA & ASTURIAS ───────────────────────────────────────────────
    'AMP': ('R', 'W', 'town', 2),  # Ampuero, small
    'ESP': ('G', 'W', 'town', 2),  # Espinosa de los Monteros, small
    'REI': ('G', 'M', 'town', 2),  # Reinosa, small
    'SAN': ('R', 'W', 'city', 3),  # Santander, 10,000
    'POT': ('G', 'W', 'town', 2),  # Potes, very small
    'VLL': ('R', 'M', 'town', 2),  # Velilla, small
    'ONT': ('R', 'W', 'town', 2),  # Unquera, small
    'INF': ('R', 'W', 'town', 2),  # Infiesto, small
    'GIJ': ('R', 'W', 'city', 3),  # Gijón, 11,800
    'OVI': ('R', 'W', 'city', 3),  # Oviedo, 13,550
    'COL': ('G', 'W', 'town', 2),  # Collanzo, very small
    'NAV': ('R', 'W', 'town', 2),  # Navia, small coastal
    'LAR': ('G', 'W', 'town', 2),  # Larón, very small

    # ── GALICIA ────────────────────────────────────────────────────────────
    'MON': ('R', 'W', 'town', 2),  # Mondoñedo, small episcopal
    'BET': ('R', 'W', 'town', 2),  # Betanzos, ~5-6k
    'FER': ('R', 'W', 'city', 3),  # Ferrol, ~15-18k est. (major naval arsenal)
    'COR': ('R', 'W', 'city', 3),  # La Coruña, 13,575
    'LUG': ('R', 'W', 'city', 3),  # Lugo, ~10-12k est. (1850=21,314)
    'STG': ('R', 'W', 'city', 3),  # Santiago de Compostela, 15,582
    'ORE': ('R', 'W', 'town', 2),  # Orense, ~6-8k est.
    'RBD': ('R', 'W', 'town', 2),  # Ribadavia, small
    'VIG': ('R', 'W', 'town', 2),  # Vigo, 2,933
    'TUY': ('R', 'W', 'town', 2),  # Tuy, small border town

    # ── OLD CASTILE & LEÓN ─────────────────────────────────────────────────
    'CRR': ('F', 'B', 'town', 2),  # Carrión de los Condes, small
    'PLC': ('F', 'B', 'city', 3),  # Palencia, 9,563
    'VAD': ('F', 'B', 'major_city', 4),  # Valladolid, 21,099
    'TDS': ('F', 'B', 'town', 2),  # Tordesillas, small
    'PEF': ('R', 'B', 'town', 2),  # Peñafiel, small
    'YSC': ('F', 'B', 'town', 2),  # Íscar, small
    'SGV': ('R', 'B', 'city', 3),  # Segovia, 9,865
    'RIA': ('R', 'B', 'town', 2),  # Riaza, small
    'BUR': ('F', 'B', 'city', 3),  # Burgos, 13,614
    'VLD': ('F', 'B', 'town', 2),  # Villadiego, small
    'MCR': ('R', 'B', 'town', 2),  # Mecerreyes, very small
    'ARN': ('F', 'B', 'town', 2),  # Aranda de Duero, 3,619
    'SOR': ('R', 'B', 'town', 2),  # Soria, 4,569
    'DEZ': ('R', 'B', 'town', 2),  # Deza, very small
    'ALM': ('R', 'B', 'town', 2),  # Almazán, small
    'BRL': ('R', 'B', 'town', 2),  # El Burgo de Osma, small
    'UTR': ('R', 'B', 'town', 2),  # Utrilla, very small
    'AST': ('F', 'M', 'town', 2),  # Astorga, ~3-4k
    'LEO': ('F', 'M', 'town', 2),  # León, 6,051 (surprisingly small)
    'ALZ': ('R', 'M', 'town', 2),  # Almanza, very small
    'VDR': ('F', 'B', 'town', 2),  # Valderas, small
    'VLF': ('R', 'W', 'town', 2),  # Villafranca del Bierzo, small
    'MTR': ('R', 'M', 'town', 2),  # Monterrei, small

    # ── NEW CASTILE / MESETA SUR ───────────────────────────────────────────
    'MAD': ('R', 'B', 'capital', 6),  # Madrid, 156,626
    'ESC': ('G', 'M', 'town', 2),  # El Escorial, small royal site
    'BTR': ('G', 'M', 'town', 2),  # Buitrago, small
    'GUA': ('R', 'B', 'town', 2),  # Guadalajara, 6,297
    'SIG': ('R', 'B', 'town', 2),  # Sigüenza, ~3k
    'MOL': ('R', 'B', 'town', 2),  # Molina de Aragón, small
    'CNV': ('R', 'B', 'town', 2),  # Cañaveras, very small
    'CUE': ('G', 'M', 'city', 3),  # Cuenca, 8,753
    'MOY': ('R', 'B', 'town', 2),  # Moya, very small
    'UTL': ('R', 'B', 'town', 2),  # Utiel, ~4,479
    'TAM': ('F', 'B', 'town', 2),  # Tarazona de la Mancha, small
    'ARJ': ('F', 'B', 'town', 2),  # Aranjuez, ~5k
    'TLD': ('R', 'B', 'city', 3),  # Toledo, 18,021
    'LGU': ('F', 'B', 'town', 2),  # La Guardia, small
    'HUT': ('R', 'B', 'town', 2),  # Huete, ~2,606
    'EDR': ('R', 'B', 'town', 2),  # Espinosa del Rey, small
    'ORC': ('R', 'B', 'town', 2),  # Orcaja, very small
    'CNS': ('F', 'B', 'town', 2),  # Consuegra, ~6,192
    'BNL': ('F', 'B', 'town', 2),  # El Bonillo, small
    'SLN': ('F', 'B', 'town', 2),  # La Solana, 5,609
    'CRL': ('F', 'B', 'city', 3),  # Ciudad Real, 8,089
    'ACR': ('R', 'B', 'town', 2),  # Alcaraz, 7,690 (just under 8k)
    'QMD': ('R', 'B', 'town', 2),  # Quemada, very small
    'AMD': ('R', 'B', 'town', 2),  # Almadén, 6,435
    'LIN': ('R', 'B', 'town', 2),  # Linares, 5,011
    'AND': ('F', 'B', 'city', 3),  # Andújar, 9,550
    'IN2': ('R', 'B', 'intersection', 1),  # Intersecció Andújar-Linares-Jaén

    # ── EXTREMADURA ────────────────────────────────────────────────────────
    'MBT': ('G', 'M', 'town', 2),  # Mombeltrán, small
    'TAL': ('F', 'B', 'town', 2),  # Talavera de la Reina, 7,818 (just under 8k)
    'PLA': ('R', 'M', 'town', 2),  # Plasencia, 4,852
    'CRI': ('F', 'B', 'town', 2),  # Coria, ~3-4k
    'TRJ': ('R', 'B', 'town', 2),  # Trujillo, 4,106
    'ABQ': ('R', 'B', 'town', 2),  # Alburquerque, 5,220
    'MER': ('F', 'B', 'town', 2),  # Mérida, 3,934
    'MDL': ('F', 'B', 'town', 2),  # Medellín, very small
    'BAD': ('F', 'B', 'city', 3),  # Badajoz, 11,872
    'PAC': ('R', 'B', 'town', 2),  # Puebla de Alcócer, small
    'AMJ': ('F', 'B', 'town', 2),  # Almendralejo, 4,230
    'ZAF': ('R', 'B', 'town', 2),  # Zafra, 5,633
    'JRC': ('R', 'B', 'town', 2),  # Jerez de los Caballeros, 7,371
    'LLR': ('R', 'B', 'town', 2),  # Llerena, 5,306

    # ── PORTUGAL ───────────────────────────────────────────────────────────
    # (no 1800 census; estimates from known history and 1850 values)
    'BRG': ('R', 'M', 'town', 2),  # Bragança, ~5-7k est.
    'MND': ('R', 'M', 'town', 2),  # Miranda do Douro, small
    'ZAM': ('F', 'B', 'city', 3),  # Zamora, 9,881
    'TRO': ('F', 'B', 'town', 2),  # Toro, 7,108
    'MRL': ('R', 'M', 'town', 2),  # Mirandela, small
    'CAM': ('F', 'M', 'town', 2),  # Caminha, small
    'BRA': ('R', 'W', 'major_city', 4),  # Braga, ~20-25k est.
    'GUI': ('R', 'W', 'city', 3),  # Guimarães, ~10-15k est.
    'OPO': ('F', 'M', 'regional_capital', 5),  # Oporto, ~70-80k est.
    'HIN': ('R', 'B', 'town', 2),  # Hinojosa del Duero, small
    'SLM': ('R', 'B', 'city', 3),  # Salamanca, ~15-18k est. (1850=15,213)
    'ALA': ('F', 'B', 'town', 2),  # Alaejos, small
    'CRO': ('R', 'M', 'town', 2),  # Ciudad Rodrigo, 5,254
    'MAG': ('F', 'B', 'town', 2),  # Madrigal de las Altas Torres, small
    'BON': ('R', 'M', 'town', 2),  # Bonilla de la Sierra, very small
    'AVI': ('R', 'B', 'town', 2),  # Ávila, 5,178
    'OVR': ('F', 'M', 'town', 2),  # Ovar, small
    'LAM': ('R', 'M', 'town', 2),  # Lamego, ~5-7k est.
    'SJP': ('R', 'M', 'town', 2),  # São João da Pesqueira, small
    'ALD': ('R', 'M', 'town', 2),  # Almeida, small fortress
    'VIS': ('R', 'M', 'town', 2),  # Viseu, ~5-8k est.
    'GRD': ('G', 'M', 'town', 2),  # Guarda, ~5-7k est.
    'CVL': ('G', 'M', 'town', 2),  # Covilhã, ~5-8k est.
    'CTB': ('R', 'M', 'town', 2),  # Castelo Branco, ~5-8k est.
    'COI': ('R', 'W', 'city', 3),  # Coimbra, ~15-20k est.
    'PTL': ('R', 'M', 'town', 2),  # Portalegre, ~5-8k est.
    'ETZ': ('F', 'B', 'town', 2),  # Estremoz, ~5k est.
    'EVO': ('F', 'B', 'city', 3),  # Évora, ~12-15k est.
    'LEI': ('R', 'M', 'city', 3),  # Leiria, ~8-12k est.
    'ABR': ('R', 'M', 'town', 2),  # Abrantes, ~6-8k est.
    'SNT': ('R', 'M', 'city', 3),  # Santarém, ~10-15k est.
    'TVD': ('R', 'M', 'town', 2),  # Torres Vedras, small
    'LIS': ('R', 'M', 'capital', 6),  # Lisboa, ~250-300k
    'AMA': ('R', 'M', 'town', 2),  # Almada, small
    'SET': ('R', 'M', 'city', 3),  # Setúbal, ~10-15k est.
    'BEJ': ('F', 'B', 'city', 3),  # Beja, ~8-10k est.
    'CTV': ('F', 'B', 'town', 2),  # Castro Verde, small
    'SRD': ('R', 'M', 'town', 2),  # Serdão, very small
    'LAG': ('F', 'M', 'town', 2),  # Lagos, ~5-8k est.
    'FAR': ('F', 'M', 'city', 3),  # Faro, ~8-12k est.

    # ── ANDALUSIA ──────────────────────────────────────────────────────────
    'JAE': ('R', 'M', 'city', 3),  # Jaén, 16,249
    'ALC': ('R', 'M', 'city', 3),  # Alcalá la Real, 11,495
    'COQ': ('R', 'B', 'town', 2),  # Conquista, very small
    'BJL': ('F', 'B', 'town', 2),  # Bujalance, 7,486
    'CDB': ('F', 'M', 'major_city', 4),  # Córdoba, 37,826
    'MTL': ('R', 'M', 'city', 3),  # Montilla, 13,979
    'LCN': ('R', 'M', 'city', 3),  # Lucena, 17,127
    'HLV': ('F', 'M', 'town', 2),  # Huelva, 5,377
    'FRJ': ('R', 'M', 'town', 2),  # Fregenal de la Sierra, 4,496
    'CST': ('G', 'W', 'town', 2),  # Constantina, 4,956
    'CRM': ('F', 'B', 'city', 3),  # Carmona, 9,911
    'ECI': ('F', 'B', 'major_city', 4),  # Écija, 29,343
    'SEV': ('F', 'M', 'regional_capital', 5),  # Sevilla, 80,915
    'CAD': ('F', 'M', 'regional_capital', 5),  # Cádiz, 71,080 (constitutional capital)
    'MDS': ('R', 'M', 'city', 3),  # Medina Sidonia, 11,338
    'TRF': ('F', 'M', 'town', 2),  # Tarifa, 7,548
    'GIB': ('F', 'M', 'town', 2),  # Gibraltar, British fortress, small
    'RON': ('G', 'M', 'city', 3),  # Ronda, 11,055
    'MRB': ('F', 'M', 'town', 2),  # Marbella, ~4,820
    'MAL': ('F', 'M', 'regional_capital', 5),  # Málaga, 51,098
    'ANT': ('R', 'M', 'major_city', 4),  # Antequera, 20,266
    'ALH': ('R', 'M', 'town', 2),  # Alhama de Granada, 6,723
    'LOJ': ('R', 'M', 'city', 3),  # Loja, 11,185
    'GRA': ('R', 'M', 'regional_capital', 5),  # Granada, 56,541
    'AMR': ('F', 'D', 'city', 3),  # Almería, 14,958
    'BAZ': ('R', 'B', 'town', 2),  # Baza, 7,720
    'HSC': ('R', 'B', 'town', 2),  # Huéscar, 6,383
    'VRA': ('F', 'D', 'city', 3),  # Vera, 8,133
    'IN3': ('R', 'M', 'intersection', 1),  # Intersecció Ronda-Marbella-Málaga
}

# ── Terrain precedence for edges ──────────────────────────────────────────────
TERRA1_RANK = {'G': 3, 'R': 2, 'F': 1, 'O': 0}
TERRA2_RANK = {'W': 4, 'M': 3, 'B': 2, 'D': 1, 'O': 0}

def edge_terrain(a, b):
    t1 = a[0] if TERRA1_RANK[a[0]] >= TERRA1_RANK[b[0]] else b[0]
    t2 = a[1] if TERRA2_RANK[a[1]] >= TERRA2_RANK[b[1]] else b[1]
    return t1, t2

def edge_road_type(imp_a, imp_b):
    """Primary if either endpoint is a city or above (importance >= 3)."""
    return 'primary' if max(imp_a, imp_b) >= 3 else 'secondary'

# ── Read source CSVs ──────────────────────────────────────────────────────────
nodes = {}
with open(HERE / 'nodes.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        nodes[row['node_id']] = {
            'full_name': row['full_name'],
            'latitude':  float(row['latitude']),
            'longitude': float(row['longitude']),
        }

edges_raw = []
with open(HERE / 'edges.csv', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        edges_raw.append((row['edge_id'], row['node1'], row['node2']))

# ── Build enriched nodes ──────────────────────────────────────────────────────
FALLBACK = ('R', 'M', 'town', 2)
missing  = []
enriched_nodes = []
for nid, nd in nodes.items():
    attrs = NODE_ATTRS.get(nid)
    if attrs is None:
        missing.append(nid)
        attrs = FALLBACK
    t1, t2, ntype, simp = attrs
    enriched_nodes.append({
        'node_id':              nid,
        'full_name':            nd['full_name'],
        'latitude':             nd['latitude'],
        'longitude':            nd['longitude'],
        'terra1':               t1,
        'terra2':               t2,
        'node_type':            ntype,
        'strategic_importance': simp,
    })

# ── Build enriched edges ──────────────────────────────────────────────────────
enriched_edges = []
for eid, n1, n2 in edges_raw:
    nd1, nd2 = nodes[n1], nodes[n2]
    dist = haversine(nd1['latitude'], nd1['longitude'],
                     nd2['latitude'], nd2['longitude'])
    a1 = NODE_ATTRS.get(n1, FALLBACK)
    a2 = NODE_ATTRS.get(n2, FALLBACK)
    t1, t2    = edge_terrain(a1, a2)
    road_type = edge_road_type(a1[3], a2[3])
    enriched_edges.append({
        'edge_id':     eid,
        'node1':       n1,
        'node2':       n2,
        'distance_km': dist,
        'terra1':      t1,
        'terra2':      t2,
        'road_type':   road_type,
    })

# ── Write output CSVs ─────────────────────────────────────────────────────────
node_fields = ['node_id','full_name','latitude','longitude',
               'terra1','terra2','node_type','strategic_importance']
with open(HERE / 'nodes.csv', 'w', newline='', encoding='utf-8') as f:
    csv.DictWriter(f, fieldnames=node_fields).writeheader()
    csv.DictWriter(f, fieldnames=node_fields).writerows(enriched_nodes)

edge_fields = ['edge_id','node1','node2','distance_km','terra1','terra2','road_type']
with open(HERE / 'edges.csv', 'w', newline='', encoding='utf-8') as f:
    csv.DictWriter(f, fieldnames=edge_fields).writeheader()
    csv.DictWriter(f, fieldnames=edge_fields).writerows(enriched_edges)

# ── Summary ───────────────────────────────────────────────────────────────────
from collections import Counter
tc = Counter(n['node_type']   for n in enriched_nodes)
rc = Counter(e['road_type']   for e in enriched_edges)
dv = [e['distance_km']        for e in enriched_edges]

print(f"\n✓  nodes.csv  — {len(enriched_nodes)} nodes written")
for k in ['capital','major_city','city','town','intersection']:
    print(f"     {k:<14}: {tc[k]:3d}")
print(f"\n✓  edges.csv  — {len(enriched_edges)} edges written")
for k in ['primary','secondary']:
    print(f"     {k:<14}: {rc[k]:3d}")
print(f"\n   Distance stats (km)  min={min(dv):.1f}  max={max(dv):.1f}  "
      f"mean={sum(dv)/len(dv):.1f}")
if missing:
    print(f"\n⚠  Nodes with no explicit attrs (fallback R/M/town/2 used): {missing}")
else:
    print("\n   All nodes matched — no fallbacks needed.")
