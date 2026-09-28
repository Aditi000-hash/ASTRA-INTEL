"""
Script to generate a synthetic, unclassified sample defence specification PDF
for testing and demonstration of ASTRA INTEL.
"""

import fitz  # PyMuPDF


def create_sample_defence_pdf(output_path: str):
    doc = fitz.open()

    # --- PAGE 1 ---
    page1 = doc.new_page()
    rect1 = fitz.Rect(50, 50, 550, 750)
    text1 = (
        "PROJECT ASTRA: AUTONOMOUS TACTICAL RECONNAISSANCE SYSTEM (ATARS)\n"
        "DEFENCE DOCUMENT INTELLIGENCE SAMPLE SPECIFICATION - UNCLASSIFIED\n"
        "Document Reference: ASTRA-SPEC-2026-V1\n"
        "Date: October 2026\n\n"
        "1.0 EXECUTIVE OVERVIEW & MISSION SCOPE\n"
        "Project Astra is an advanced unmanned aerial reconnaissance platform engineered to deliver "
        "persistent intelligence, surveillance, target acquisition, and reconnaissance (ISTAR) "
        "capabilities across contested multi-domain theatres.\n\n"
        "The system addresses the operational requirement for deep-penetration tactical awareness "
        "without exposing human crew to integrated air defence systems (IADS).\n\n"
        "1.1 SYSTEM CONCEPT & OPERATIONAL ARCHITECTURE\n"
        "The ATARS airframe utilizes carbon-fiber composite materials offering a low radar cross-section (RCS). "
        "Equipped with an autonomous edge-compute processing node, the platform performs real-time automated "
        "target recognition (ATR) directly on-board, minimizing transmission bandwidth back to ground control."
    )
    page1.insert_textbox(rect1, text1, fontsize=11, fontname="helv")

    # --- PAGE 2 ---
    page2 = doc.new_page()
    rect2 = fitz.Rect(50, 50, 550, 750)
    text2 = (
        "PROJECT ASTRA: TECHNICAL SPECIFICATIONS & AVIONICS\n"
        "Document Reference: ASTRA-SPEC-2026-V1 | Page 2\n\n"
        "2.0 PROPULSION & ENDURANCE\n"
        "- Powerplant: Hybrid hydrogen-electric fuel cell with regenerative turbo-alternator.\n"
        "- Operational Flight Endurance: 36 hours continuous loiter time.\n"
        "- Maximum Service Ceiling: 45,000 feet above mean sea level (MSL).\n"
        "- Cruising Speed: 180 knots (approx 333 km/h).\n"
        "- Maximum Dash Speed: 260 knots.\n\n"
        "2.1 SECURE COMMUNICATIONS & DATA LINKS\n"
        "The system maintains communications through redundant line-of-sight (LOS) and satellite communications (SATCOM):\n"
        "- Ku-Band and Ka-band secure satellite uplinks with automated anti-jamming beam steering.\n"
        "- Encryption: Hardware-level AES-256-GCM authenticated telemetry.\n"
        "- Frequency-hopping spread spectrum (FHSS) operating between 12 GHz and 18 GHz."
    )
    page2.insert_textbox(rect2, text2, fontsize=11, fontname="helv")

    # --- PAGE 3 ---
    page3 = doc.new_page()
    rect3 = fitz.Rect(50, 50, 550, 750)
    text3 = (
        "PROJECT ASTRA: TACTICAL PAYLOADS & MAJOR APPLICATIONS\n"
        "Document Reference: ASTRA-SPEC-2026-V1 | Page 3\n\n"
        "3.0 MAJOR APPLICATIONS\n"
        "The Astra system is configured to fulfill four primary operational defence roles:\n"
        "1. Maritime Border Surveillance: Continuous monitoring of littoral waters and identification of non-flagged surface vessels.\n"
        "2. Artillery Fire Direction: Rapid forward observer targeting and battle damage assessment (BDA).\n"
        "3. Tactical Electronic Warfare (EW): Signals intelligence (SIGINT) emitter detection and localized GPS denial protection.\n"
        "4. Search and Rescue (SAR): High-resolution infrared scanning during severe weather operations.\n\n"
        "3.1 SENSOR ARCHITECTURE\n"
        "- High-Definition Electro-Optical/Infrared (EO/IR) gyrostabilized turret with 50x optical magnification.\n"
        "- Multi-mode Synthetic Aperture Radar (SAR) capable of ground moving target indication (GMTI).\n"
        "- Long-range LiDAR scanner for sub-meter 3D terrain topography mapping."
    )
    page3.insert_textbox(rect3, text3, fontsize=11, fontname="helv")

    # --- PAGE 4 ---
    page4 = doc.new_page()
    rect4 = fitz.Rect(50, 50, 550, 750)
    text4 = (
        "PROJECT ASTRA: ENVIRONMENTAL CONSTRAINTS & DEPLOYMENT\n"
        "Document Reference: ASTRA-SPEC-2026-V1 | Page 4\n\n"
        "4.0 ENVIRONMENTAL & WEATHER LIMITATIONS\n"
        "- Operating Ambient Temperature Range: -40 degrees Celsius to +55 degrees Celsius.\n"
        "- Maximum Crosswind Landing Limit: 35 knots.\n"
        "- De-icing System: Electro-thermal leading edge wing heating.\n\n"
        "4.1 LOGISTICS & OPERATIONAL TURNAROUND\n"
        "- Base Turnaround Cycle: 90 minutes between recovery and redeployment.\n"
        "- Mean Time Between Failures (MTBF): Certified at 1,200 operational flight hours.\n"
        "- Crew Requirement: 2 operators per ground control station (one mission commander and one payload specialist)."
    )
    page4.insert_textbox(rect4, text4, fontsize=11, fontname="helv")

    doc.save(output_path)
    doc.close()
    print(f"Sample PDF created successfully at: {output_path}")


if __name__ == "__main__":
    create_sample_defence_pdf("sample_docs/astra_defence_spec.pdf")
