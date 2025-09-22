# Nadel-Detektion mit ArUco-Referenz

Dieses Projekt beschäftigt sich mit der automatisierten Erkennung und 3D-Lokalisierung einer Nadelspitze in medizinischen Testbildern.  
Als Referenz dient ein auf dem Pad angebrachter **ArUco-Marker**, der die Weltkoordinaten definiert.  
Die Pipeline kombiniert klassische Bildverarbeitung (ROI-Maskierung, Template-Matching, Feature-Matching) mit Kamerakalibrierung und 3D-Geometrie.

---

## Ziele
- **Automatische Erkennung** der Nadel in Bildern oder Live-Kamerafeeds.
- **Bestimmung der Nadelspitze** im 2D-Bild.
- **Projektion der Spitze in 3D** (Weltkoordinaten relativ zum ArUco-Marker).
- **Berechnung der Distanz** zwischen Marker-Zentrum und Nadelspitze.
- Visualisierung der Ergebnisse mit Overlays (ArUco-Position, Vektor Marker→Nadelspitze, Distanzbeschriftung).

---

## Pipeline

### 1. Kamera-Kalibrierung
- Die Kamera ist eine Fisheye-Kamera.
- Vorverarbeitung: Entzerrung (Rectification) → Erzeugung eines pinhole-ähnlichen Bildes mit `cv.fisheye.undistortImage`.
- Neue Intrinsikmatrix `newK` wird gespeichert und in allen nachfolgenden Schritten verwendet.

### 2. ArUco-Detektion
- Erkennung eines **4x4 ArUco-Markers** im entzerrten Bild.
- Schätzung von Rotation `rvec` und Translation `tvec` mittels `cv.aruco.estimatePoseSingleMarkers`.
- Aufbau der Transformationen:
  - Marker → Kamera (T_mc)
  - Kamera → Marker/Welt (T_cw)
- Berechnung der **Kameraposition** in Weltkoordinaten (Marker-Frame).
- Visualisierung: Marker-Umrandung + Achsen.

### 3. ROI-Maskierung
- Extraktion des relevanten Bereichs (Pad mit Silikonhaut).
- HSV-Schwellenwerte + Morphologie-Operationen filtern Hintergrund und Vignettierung.
- Ergebnis: Binäre Maske der ROI.

### 4. Template- und Feature-Matching
- Vorbereitung:
  - Croppen von Nadel-Patches aus Trainingsbildern (`templates/`).
  - Optional: Masken und Markierung der Nadelspitze im Template.
  - Extraktion von Deskriptoren (SIFT, AKAZE, ORB).
- Laufzeit:
  - ROI wird auf Templates gematcht.
  - Homographie oder affine Transformation (RANSAC) wird geschätzt.
  - Die gespeicherte Nadelspitze im Template wird mittels Transformation ins Szenenbild projiziert → `tip_px_scene`.

### 5. 3D-Projektion der Nadelspitze
- Annahme: Die Nadel liegt in einer Ebene parallel zum Marker, mit festem Offset `Δz` (z. B. 29 mm).
- Vorgehen:
  - Berechnung des Kamerarays durch Pixelkoordinate `tip_px_scene`.
  - Schnitt des Rays mit der Marker-Ebene (oder verschobenen Ebene).
  - Ergebnis: 3D-Koordinate `X_w` der Nadelspitze in Weltkoordinaten.

### 6. Distanzberechnung
- Abstand der Spitze zum Marker-Ursprung:
  \[
  d = \|X_w - (0,0,0)\|
  \]
- Darstellung in Millimetern.

### 7. Visualisierung
- Anzeige des Vektors von Markerzentrum → Nadelspitze.
- Beschriftung mit Distanzwert.
- Nebeneinanderstellung von ROI-Maske und Overlay.

---

## Beispielausgabe
- Camera pos (m): [-0.14 0.18 0.41]
- Projection Error: (0.72, 0.024)
- Tip world: [0.012, -0.025, 0.029] | distance: 31.4 mm

Bildebene:  
- Grün: ArUco-Achsen  
- Gelb: Vektor Markerzentrum → Nadelspitze  
- Rot: Nadelspitze  

---

## Nutzung

### Statischer Modus (Bilder)
```
bash
python main.py --static --images "test_images/*.jpg"
```
- Iteriert über alle Bilder im Ordner test_images.
- Wartet pro Bild auf Tastendruck (q zum Abbrechen).

### Live-Modus (Kamera)
```
bash
python main.py --live --camera 0
```

- Nutzt angeschlossene Kamera (Index 0).
- Echtzeit-Erkennung mit Anzeige der Overlays.
- q oder Esc beendet, p pausiert/fortsetzt.

### Verzeichnisstruktur
```
project/
│
├── camera.py              # Kamera-Klasse mit Kalibrierung und Undistortion
├── needle_detect.py       # Hauptlogik ArUco + Nadel-3D
├── patchmask.py           # ROI-Maskierung und Template-Handling
├── templates/             # Gespeicherte Nadel-Templates (+ Masken + Metadaten)
├── test_images/           # Testbilder mit Nadeln + Marker
└── README.md              # (dieses Dokument)
```

## Bekannte Einschränkungen

- Nadel liegt nicht exakt in Marker-Ebene → Offset muss manuell eingestellt werden.
- Feature-Matching empfindlich gegenüber Beleuchtung und Reflexionen.
- Genauigkeit hängt stark von Kalibrierung und Marker-Detektion ab.

## Mögliche Erweiterungen

- Multi-View Fusion (mehrere Kamerapositionen → triangulierte Spitze).
- Deep Learning für robustere Nadeldetektion.
- Automatische Schätzung des Offsets Δz durch Kalibrierung.
- Integration in ein Echtzeit-Tracking-Framework.