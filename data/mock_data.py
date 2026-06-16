# data/mock_data.py
import random

# A predefined set of high-fidelity clinical note paragraphs representing a diverse set of patient histories.
# This simulates a subset of the 50,000 discharge summaries.

MOCK_PATIENTS_DATA = [
    # ========================== PATIENT PT-8829 (Mild Angina, Stent, Stable) ==========================
    {
        "text": "Patient PT-8829, a 62-year-old male, was admitted for elective cardiac catheterization following an abnormal myocardial perfusion stress test showing mild reversible ischemia in the anterior wall.",
        "patient_id": "PT-8829",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "During the procedure, Dr. Suresh identified a 75% stenosis in the mid-left anterior descending (LAD) coronary artery. A drug-eluting stent (DES) was successfully deployed with 0% residual stenosis and TIMI 3 flow.",
        "patient_id": "PT-8829",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "Post-procedure follow-up notes for PT-8829 indicate stable cardiac status. The patient reports occasional mild chest tightness, which is atypical and relieved by rest, with no signs of exertional dyspnea.",
        "patient_id": "PT-8829",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "The discharge medication regimen for patient PT-8829 includes Aspirin 75mg once daily, Clopidogrel 75mg once daily (for dual antiplatelet therapy), Atorvastatin 40mg at bedtime, and Ramipril 2.5mg daily.",
        "patient_id": "PT-8829",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "For patient PT-8829, an electrocardiogram (ECG) performed during the follow-up clinic showed normal sinus rhythm at 65 bpm with no ST-segment elevations or T-wave inversions, confirming no active ischemia.",
        "patient_id": "PT-8829",
        "doctor_name": "Dr. Ramesh"
    },

    # ========================== PATIENT PT-1234 (Severe Cardiac Arrest, CABG, ICD) ==========================
    {
        "text": "Patient PT-1234 presented to the emergency department via EMS in acute cardiogenic shock following a severe, crushing sub-sternal chest pain radiating to the left arm and jaw.",
        "patient_id": "PT-1234",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "During initial evaluation, patient PT-1234 went into ventricular fibrillation and suffered a sudden cardiac arrest. Cardiopulmonary resuscitation (CPR) was initiated, and the patient was successfully defibrillated back to sinus rhythm after three 200J biphasic shocks.",
        "patient_id": "PT-1234",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "Coronary angiography for PT-1234 revealed severe triple-vessel CAD. Dr. Suresh requested an urgent consult from cardiothoracic surgery, and the patient was taken to the OR for an emergent Coronary Artery Bypass Graft (CABG) x3 (LIMA-LAD, SVG-OM1, SVG-RCA).",
        "patient_id": "PT-1234",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "The post-operative ICU course for patient PT-1234 was highly complicated by cardiogenic shock requiring intra-aortic balloon pump (IABP) support and high-dose inotropes (epinephrine and dobutamine).",
        "patient_id": "PT-1234",
        "doctor_name": "Dr. Ramesh"
    },
    {
        "text": "Given the history of sudden cardiac arrest due to ventricular fibrillation and a post-infarction left ventricular ejection fraction (LVEF) of 25%, patient PT-1234 underwent successful implantation of a dual-chamber Implantable Cardioverter-Defibrillator (ICD) for secondary prevention prior to discharge.",
        "patient_id": "PT-1234",
        "doctor_name": "Dr. Ramesh"
    },

    # ========================== PATIENT PT-5566 (Type-2 Diabetes, Diabetic Foot Ulcer) ==========================
    {
        "text": "Patient PT-5566, a 54-year-old female, was admitted due to a non-healing neuropathic ulcer on the plantar surface of the right metatarsal head, complicated by localized cellulitis.",
        "patient_id": "PT-5566",
        "doctor_name": "Dr. Ramesh"
    },
    {
        "text": "Lab results for patient PT-5566 showed a severely elevated Hemoglobin A1c of 10.4%, indicating poorly controlled Type-2 Diabetes Mellitus. Nephropathy screening was positive for microalbuminuria.",
        "patient_id": "PT-5566",
        "doctor_name": "Dr. Ramesh"
    },
    {
        "text": "Surgical debridement of the right foot ulcer was performed for PT-5566. Bone biopsy was negative for osteomyelitis, and wound cultures grew Methicillin-Sensitive Staphylococcus Aureus (MSSA), treated with IV Cefazolin.",
        "patient_id": "PT-5566",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "To optimize glycemic control for patient PT-5566, her outpatient oral medications were discontinued, and she was transitioned to an intensive basal-bolus insulin regimen consisting of Insulin Glargine (Lantus) and rapid-acting Insulin Aspart (Novolog).",
        "patient_id": "PT-5566",
        "doctor_name": "Dr. Ramesh"
    },

    # ========================== PATIENT PT-7700 (COPD, Asthma, Respiratory Failure) ==========================
    {
        "text": "Patient PT-7700, an 68-year-old female with a 40 pack-year smoking history, was admitted with an acute exacerbation of Chronic Obstructive Pulmonary Disease (COPD) and acute respiratory failure.",
        "patient_id": "PT-7700",
        "doctor_name": "Dr. Ramesh"
    },
    {
        "text": "Arterial blood gas (ABG) on admission for PT-7700 showed severe respiratory acidosis and hypoxemia. The patient was started on non-invasive positive pressure ventilation (BiPAP) in the respiratory intermediate care unit.",
        "patient_id": "PT-7700",
        "doctor_name": "Dr. Ramesh"
    },
    {
        "text": "Pharmacotherapy for PT-7700's COPD flare-up included intravenous Methylprednisolone, frequent nebulizer treatments with Albuterol/Ipratropium (Duoneb), and empiric antibiotics with Azithromycin.",
        "patient_id": "PT-7700",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "At discharge, patient PT-7700 was stable on room air. She was prescribed home oxygen at 2L/min via nasal cannula, a tapering oral prednisone course, and was referred to pulmonary rehabilitation.",
        "patient_id": "PT-7700",
        "doctor_name": "Dr. Ramesh"
    },

    # ========================== PATIENT PT-9900 (Chronic Kidney Disease, Hemodialysis) ==========================
    {
        "text": "Patient PT-9900 is a 71-year-old male with Stage 5 Chronic Kidney Disease (CKD) secondary to long-standing hypertensive nephrosclerosis, admitted for initiation of maintenance hemodialysis.",
        "patient_id": "PT-9900",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "A left forearm arteriovenous (AV) fistula was surgically created for PT-9900 by vascular surgery. Until the fistula matures, a temporary tunneled dialysis catheter was placed in the right internal jugular vein.",
        "patient_id": "PT-9900",
        "doctor_name": "Dr. Suresh"
    },
    {
        "text": "For patient PT-9900, dialysis-related lab work revealed renal anemia with a hemoglobin of 8.2 g/dL. Treatment was initiated with Epoetin alfa (Epogen) and intravenous iron sucrose supplementation.",
        "patient_id": "PT-9900",
        "doctor_name": "Dr. Ramesh"
    }
]

def get_all_mock_documents():
    """
    Returns a list of document objects or dictionaries.
    To simulate a larger dataset of 50,000, we can replicate or keep these core distinct patient stories.
    We return these rich clinical records.
    """
    # Let's also dynamically generate a set of filler records for other patients to show scale
    documents = list(MOCK_PATIENTS_DATA)
    
    # Generate 50 additional clinical records for filler patients to make similarity search more interesting
    filler_patients = [f"PT-{random.randint(2000, 8000)}" for _ in range(10)]
    doctors = ["Dr. Suresh", "Dr. Ramesh", "Dr. Verma", "Dr. Mehta"]
    conditions = [
        ("hypertension", "The patient's blood pressure was noted to be elevated at 145/95 mmHg. Recommended lifestyle modifications and started Lisinopril 10mg daily."),
        ("hyperlipidemia", "Lipid panel showed an elevated LDL cholesterol of 165 mg/dL. Prescribed Atorvastatin 20mg daily and dietary consulting."),
        ("gastroesophageal reflux", "Reports persistent heartburn and acid regurgitation. Started Omeprazole 20mg daily before breakfast and advised avoiding late meals."),
        ("osteoarthritis", "Complains of bilateral knee pain, worse with weight-bearing. X-rays show moderate joint space narrowing. Prescribed Acetaminophen and physical therapy."),
        ("hypothyroidism", "Routine labs showed a TSH of 6.2 mIU/L with low free T4. Initiated Levothyroxine 50mcg daily with repeat labs in 6 weeks.")
    ]
    
    for i in range(50):
        p_id = random.choice(filler_patients)
        doc = random.choice(doctors)
        cond_name, cond_text = random.choice(conditions)
        documents.append({
            "text": f"Patient {p_id} clinical update: {cond_text}",
            "patient_id": p_id,
            "doctor_name": doc
        })
        
    return documents
