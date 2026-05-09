"""
concerns.py
===========
Regex keyword patterns for customer skin and health concerns.

Exports:
    CONCERN_KEYWORDS — Dict mapping concern label (e.g. "Acne-Prone Skin")
                       to a list of regex patterns matched against chat text.
                       Used by signal_detectors.py and ml_inference.py to
                       tag sessions with relevant concern labels.
"""

import re

CONCERN_KEYWORDS = {
    # ── SKINCARE CONCERNS ──────────────────────────────────────────────────
    "Acne-Prone Skin": [
        r"\bacne\b", r"\bpimples?\b", r"\bbreakouts?\b", r"\bwhiteheads?\b",
        r"\bblackheads?\b", r"\bzits?\b", r"\bcyst(ic)?\b", r"\bspots?\b",
        r"\bblemish(es)?\b", r"\bpore\s*clog(ging)?\b"
    ],
    "Aging Skin": [
        r"\baging\b", r"\bmature\s*skin\b", r"\bage\s*spots?\b", r"\bsagging\b",
        r"\bsigns\s*of\s*aging\b"
    ],
    "Anti-aging": [
        r"\banti[- ]?aging\b", r"\brejuvenat(e|ion)\b",
        r"\brenew(al)?\b", r"\bage\s*defying\b"
    ],
    "Brightening": [
        r"\bbrighten(ing)?\b", r"\bradiance\b", r"\blighten(ing)?\b",
        r"\bglow\b", r"\bwhitening\b"
    ],
    "Combination Skin": [
        r"\bcombination\s*skin\b", r"\boily\s*and\s*dry\b"
    ],
    "Cracked Skin": [
        r"\bcrack(ed)?\b", r"\bchapped\b", r"\bfissures?\b", r"\bcracked\s*heels?\b"
    ],
    "Damaged Skin": [
        r"\bdamaged\s*skin\b", r"\bskin\s*barrier\b", r"\brepair\s*skin\b"
    ],
    "Dark Eye Circles": [
        r"\bdark\s*circles?\b", r"\bunder\s*eye\b", r"\beye\s*bags?\b"
    ],
    "Dry Skin": [
        r"\bdry\s*skin\b", r"\bflaky\b", r"\bdryness\b", r"\bdehydrated\b",
        r"\bpeeling\b", r"\bashey\b", r"\bashy\b"
    ],
    "Dull Skin": [
        r"\bdull\b", r"\bdullness\b", r"\blusterless\b", r"\btired\s*skin\b"
    ],
    "Glowing Skin": [
        r"\bglowing\b", r"\bglow\b", r"\bglass\s*skin\b", r"\bluminous\b"
    ],
    "Hyperpigmentation": [
        r"\bhyper[- ]?pigmentation\b", r"\bdark\s*(spot|mark|patch)s?\b",
        r"\b(un)?even\s*skin\b", r"\bpigmentation\b", r"\bmelasma\b",
        r"\bdiscoloration\b", r"\bpost[- ]?acne\b"
    ],
    "Irritated Skin": [
        r"\birritat(ed|ion)\b", r"\binflamed\b", r"\bsooth(e|ing)\b",
        r"\bcalm(ing)?\b"
    ],
    "Itchy Skin": [
        r"\bitch(y|ing)?\b", r"\bscratchy\b", r"\bpruritus\b"
    ],
    "Normal Skin": [
        r"\bnormal\s*skin\b", r"\bbalanced\s*skin\b"
    ],
    "Oily Skin": [
        r"\boily\b", r"\bgreasy\b", r"\bsebum\b",
        r"\blarge\s*pores\b", r"\boiliness\b", r"\bshiny\s*skin\b"
    ],
    "Red Skin": [
        r"\bred\s*skin\b", r"\bredness\b", r"\bflush(ed)?\b", r"\brosacea\b",
        r"\berythema\b"
    ],
    "Rough & Bumpy Skin": [
        r"\brough\b", r"\bbumpy\b", r"\btextured\b", r"\bstrawberry\s*legs\b"
    ],
    "Sensitive Skin": [
        r"\bsensitive\b", r"\breactive\b", r"\bmild\s*skin\b",
        r"\bskin\s*sensitivity\b"
    ],
    "Uneven Skin": [
        r"\buneven\b", r"\btexture\b", r"\bblotchy\b"
    ],
    "Wrinkles & Fine Lines": [
        r"\bwrinkles?\b", r"\bfine\s*lines?\b", r"\bcrow\s*feet\b", r"\bcreases?\b"
    ],
    "Eczema": [
        r"\beczema\b", r"\bdermatitis\b", r"\batopic\b"
    ],
    "Psoriasis": [
        r"\bpsoriasis\b", r"\bpsoriatic\b",
        r"\bskin\s*plaques?\b", r"\bsilver\s*scales?\b"
    ],
    "Keratosis Pilaris": [
        r"\bkeratosis\s*pilaris\b", r"\bchicken\s*skin\b",
        r"\bbumps\s*on\s*(arms?|legs?|thighs?)\b"
    ],

    # ── HAIR & SCALP CONCERNS ──────────────────────────────────────────────
    "Hair Health": [
        r"\bhair\s*loss\b", r"\bthinning\b", r"\bhair\s*growth\b",
        r"\bbalding\b", r"\bweak\s*hair\b", r"\bhair\s*fall\b", r"\bbreakage\b",
        r"\bshampoo\b", r"\bconditioner\b", r"\bdetangler\b",
        r"\breceding\s*hairline\b", r"\balopecia\b"
    ],
    "Curl Activator": [
        r"\bcurl(s|y)?\b", r"\bactivator\b", r"\bcoils?\b",
        r"\bdefining\s*cream\b", r"\bleave[- ]?in\b", r"\btwist[- ]?out\b"
    ],
    "Dandruff": [
        r"\bdandruff\b", r"\bscalp\s*flakes?\b", r"\bt/?gel\b", r"\bwhite\s*flakes\b"
    ],
    "Itchy Scalp": [
        r"\bitchy\s*scalp\b", r"\bscalp\s*irritation\b", r"\bscalp\s*itch\b"
    ],

    # ── WOMEN'S HEALTH & PREGNANCY ─────────────────────────────────────────
    "Pregnancy Support": [
        r"\bpregnan(t|cy)\b", r"\bmaternity\b", r"\bexpecting\b",
        r"\bmorning\s*sickness\b"
    ],
    "Prenatal Support": [
        r"\bprenatal\b", r"\bfolic\s*acid\b", r"\bpre[- ]?natal\b"
    ],
    "Postnatal Support": [
        r"\bpostnatal\b", r"\bpostpartum\b", r"\bpost[- ]?natal\b",
        r"\bafter\s*delivery\b", r"\bpost\s*birth\b", r"\bafter\s*birth\b"
    ],
    "Breastfeeding Support": [
        r"\bbreastfeeding\b", r"\bnursing\b", r"\blactation\b", r"\bmilk\s*supply\b",
        r"\bbreast\s*milk\b"
    ],
    "Menopause": [
        r"\bmenopause\b", r"\bhot\s*flashes?\b", r"\bnight\s*sweats?\b",
        r"\bperi[- ]?menopause\b"
    ],
    "Period Cramps": [
        r"\bperiod\s*cramps?\b", r"\bperiod\s*pains?\b", r"\bmenstrual\b",
        r"\bdysmenorrhea\b", r"\bpainful\s*period\b",
        r"\bmenstrual\s*cramps?\b"
    ],
    "PCOS": [
        r"\bpcos\b", r"\bpolycystic\b", r"\bovarian\s*syndrome\b"
    ],
    "Hormonal Balance": [
        r"\bhormonal\b", r"\bhormones?\b", r"\bhormone\s*imbalance\b"
    ],
    "Vaginal Health": [
        r"\bvaginal\b", r"\byeast\s*infection\b", r"\bvaginosis\b",
        r"\bvaginal\s*odou?r\b", r"\bph\s*balance\b", r"\bthrush\b"
    ],
    "UTI Treatment": [
        r"\buti\b", r"\burinary\s*tract\b", r"\bcranberry\b",
        r"\bpainful\s*urination\b", r"\bburning\s*urination\b"
    ],
    "Fertility Support": [
        r"\bfertility\b", r"\btrying\s*to\s*conceive\b", r"\bconceive\b",
        r"\bovulation\b"
    ],

    # ── MEN'S HEALTH ───────────────────────────────────────────────────────
    "Male Fertility": [
        r"\bmale\s*fertility\b", r"\bsperm\s*count\b", r"\bmotility\b"
    ],
    "Prostate Health": [
        r"\bprostate\b", r"\bprostatitis\b",
        r"\benlarged\s*prostate\b"
    ],
    "Testosterone Support": [
        r"\btestosterone\b",
        r"\blow\s*testosterone\b", r"\blibido\b", r"\bmanhood\b"
    ],
    "Razor Bumps": [
        r"\brazor\s*bumps?\b", r"\bshaving\s*bumps?\b", r"\bingrown\s*hairs?\b"
    ],
    "Sexual Health": [
        r"\bsexual\b", r"\berectile\b", r"\blibido\b", r"\bsex\s*drive\b",
        r"\bcondoms?\b"
    ],

    # ── BABY & CHILD CARE ──────────────────────────────────────────────────
    "Baby Development": [
        r"\bbaby\b", r"\binfant\b", r"\btoddler\b", r"\bchild\s*development\b"
    ],
    "Diaper Rash": [
        r"\bdiaper\s*rash\b", r"\bnappy\s*rash\b", r"\bbum\s*cream\b",
        r"\bsudocrem\b"
    ],

    # ── DIGESTIVE & GUT HEALTH ─────────────────────────────────────────────
    "Digestive Support": [
        r"\bdigest(ion|ive)\b", r"\bstomach\s*upset\b", r"\bindigestion\b"
    ],
    "Gut Health": [
        r"\bgut\b", r"\bprobiotics?\b", r"\bmicrobiome\b"
    ],
    "Bloating": [
        r"\bbloating\b", r"\bbloated\b", r"\bswollen\s*tummy\b"
    ],
    "Gas & Bloating Relief": [
        r"\bflatulence\b", r"\bfarting\b", r"\btummy\s*gas\b",
        r"\bexcess\s*gas\b", r"\bgassy\b"
    ],
    "Constipation": [
        r"\bconstipat(ed|ion)\b", r"\bhard\s*stool\b", r"\blaxative\b",
        r"\bcant\s*poop\b", r"\bno\s*bowel\b"
    ],
    "Diarrhea": [
        r"\bdiarrhea\b", r"\brunning\s*stomach\b", r"\bloose\s*stool\b",
        r"\bbowel\s*movement\b"
    ],
    "Heartburn": [
        r"\bheartburn\b", r"\bacid\s*reflux\b", r"\bgerd\b", r"\bchest\s*burning\b"
    ],
    "Antacids": [
        r"\bantacids?\b", r"\beno\b", r"\bgaviscon\b", r"\brelcer\b"
    ],
    "Ulcers": [
        r"\bulcers?\b", r"\bstomach\s*ulcers?\b", r"\bpeptic\b"
    ],

    # ── PAIN, BONES & MUSCLES ──────────────────────────────────────────────
    "Pain Relief": [
        r"\bpain(killers?)?\b", r"\bache(s)?\b", r"\bhurt(s|ing)?\b",
        r"\bparacetamol\b", r"\bibuprofen\b", r"\bpanadol\b", r"\bmara\s*moja\b"
    ],
    "Joint & Bone Health": [
        r"\bjoint(s)?\b", r"\bbone(s)?\b", r"\barthritis\b", r"\bknees?\b",
        r"\bcalcium\b", r"\bcartilage\b"
    ],
    "Muscle Cramps": [
        r"\bmuscle\s*cramps?\b", r"\bspasms?\b", r"\bcharley\s*horse\b"
    ],
    "Muscle Function": [
        r"\bmuscle\s*recovery\b", r"\bmuscle\s*repair\b",
        r"\belectrolytes?\b", r"\bmuscle\s*soreness\b"
    ],
    "Lower Back Pain": [
        r"\bback\s*pain\b", r"\bbackache\b", r"\blumbago\b", r"\bspine\b"
    ],
    "Plantar Fasciitis": [
        r"\bplantar\s*fasciitis\b", r"\bheel\s*pain\b", r"\bfoot\s*arch\b"
    ],
    "Bunions": [
        r"\bbunions?\b", r"\btoe\s*joint\b"
    ],

    # ── COLD, FLU & RESPIRATORY ────────────────────────────────────────────
    "Cold & Flu": [
        r"\bcommon\s*cold\b", r"\bflu\b", r"\bcough\b", r"\bsneeze\b",
        r"\bsore\s*throat\b", r"\bfever\b", r"\brunny\s*nose\b", r"\bchills\b"
    ],
    "Nasal Congestion": [
        r"\bcongest(ed|ion)\b", r"\bblocked\s*nose\b", r"\bstuffy\s*nose\b",
        r"\bsinus(es)?\b"
    ],
    "Respiratory Conditions": [
        r"\brespiratory\b", r"\bbreathing\b", r"\blungs?\b", r"\bchest\s*infection\b"
    ],
    "Asthma": [
        r"\basthma\b", r"\binhalers?\b", r"\bwheezing\b", r"\bventolin\b"
    ],
    "Lung Health": [
        r"\blung\s*health\b", r"\bclear\s*lungs\b"
    ],

    # ── MENTAL HEALTH, SLEEP & ENERGY ──────────────────────────────────────
    "Mental Health": [
        r"\bmental\s*health\b", r"\bdepression\b", r"\bdepressed\b"
    ],
    "Anxiety": [
        r"\banxiety\b", r"\banxious\b", r"\bpanic\b", r"\bnerves\b", r"\bnervous\b"
    ],
    "Stress Relief": [
        r"\bstress(ed)?\b", r"\btension\b", r"\bcalming\b", r"\brelax(ation)?\b"
    ],
    "Mood Support": [
        r"\bmood\b", r"\buplift\b", r"\bmood\s*swings?\b"
    ],
    "Sleep": [
        r"\bsleep\b", r"\binsomnia\b", r"\bmelatonin\b", r"\bcant\s*sleep\b",
        r"\bcan't\s*sleep\b", r"\btrouble\s*sleeping\b"
    ],
    "Snoring": [
        r"\bsnoring\b", r"\bsnore\b", r"\bsleep\s*apnea\b"
    ],
    "Fatigue": [
        r"\bfatigue\b", r"\btired(ness)?\b", r"\bexhaust(ed|ion)\b",
        r"\bweak(ness)?\b", r"\blow\s*energy\b"
    ],
    "Energy Booster": [
        r"\benergy\b", r"\bboost\b", r"\bvitality\b", r"\bstamina\b"
    ],
    "Cognitive and Memory Function": [
        r"\bmemory\b", r"\bfocus\b", r"\bconcentration\b", r"\bcognitive\b",
        r"\balzheimers\b", r"\bdementia\b"
    ],
    "Brain Health": [
        r"\bbrain\b", r"\bomega\s*3\b", r"\bdha\b"
    ],
    "Brain Development": [
        r"\bbrain\s*development\b", r"\bchild\s*brain\b"
    ],

    # ── CHRONIC & GENERAL WELLNESS ─────────────────────────────────────────
    "Blood Pressure Support": [
        r"\bblood\s*pressure\b", r"\bhypertension\b", r"\bhigh\s*bp\b", r"\blow\s*bp\b"
    ],
    "Blood Sugar Support": [
        r"\bblood\s*sugar\b", r"\bglucose\b"
    ],
    "Diabetic Support": [
        r"\bdiabet(es|ic)\b", r"\binsulin\b"
    ],
    "Cardiovascular Health": [
        r"\bcardio(vascular)?\b", r"\bheart\b"
    ],
    
    "Cholesterol": [
        r"\bcholesterol\b", r"\blipids\b", r"\bldl\b", r"\bhdl\b"
    ],
    "Liver Support": [
        r"\bliver\b", r"\bhepatitis\b", r"\bcirrhosis\b"
    ],
    "Thyroid Function": [
        r"\bthyroid\b", r"\bhypothyroid\b", r"\bhyperthyroid\b"
    ],
    "Immunity Support": [
        r"\bimmun(e|ity)\b", r"\bvitamin\s*c\b", r"\bzinc\b", r"\bdefense\b"
    ],
    "Nutritional Support": [
        r"\bnutrition(al)?\b", r"\bsupplements?\b", r"\bmultivitamin\b", r"\bvitamins?\b"
    ],
    "Weight Management": [
        r"\bweight\s*loss\b", r"\blose\s*weight\b", r"\bfat\s*burner\b",
        r"\bdiet\b", r"\bweight\s*gain\b"
    ],
    "Detoxification": [
        r"\bdetox(ification)?\b", r"\bcleanse\b"
    ],
    "Anemia": [
        r"\banemia\b", r"\banemic\b", r"\biron\s*deficiency\b", r"\blow\s*iron\b",
        r"\bblood\s*builder\b"
    ],
    "Athletic Performance": [
        r"\bathletic\b", r"\bworkout\b", r"\bpre[- ]?workout\b", r"\bsports\b",
        r"\bgym\b", r"\bcreatine\b", r"\bwhey\b"
    ],

    # ── WOUNDS, SKIN CONDITIONS & MISC ─────────────────────────────────────
    "Burns & Scalds": [
        r"\bburn(s|t)?\b", r"\bscald(s|ed)?\b", r"\bblister(s)?\b", r"\bsunburn\b"
    ],
    "Antiseptic": [
        r"\bantiseptic\b", r"\bdisinfect\b", r"\bwounds?\b", r"\bcuts?\b", r"\bdettol\b"
    ],
    "Antifungal": [
        r"\bantifungal\b", r"\bfung(us|al)\b", r"\bringworm\b", r"\bathletes\s*foot\b"
    ],
    "Anti-Inflammatory": [
        r"\banti[- ]?inflammatory\b", r"\binflammation\b", r"\bswelling\b"
    ],
    "Antioxidant": [
        r"\bantioxidants?\b", r"\bfree\s*radicals?\b"
    ],
    "Collagen Production": [
        r"\bcollagen\b", r"\belasticity\b"
    ],
    "Firming & Lifting": [
        r"\bfirming\b", r"\blifting\b", r"\btighten(ing)?\b"
    ],
    "Eye & Vision Support": [
        r"\beye(s)?\b", r"\bvision\b", r"\bsight\b", r"\bblur(red)?\b", r"\bcataracts?\b"
    ],
    "Puffy Eyes": [
        r"\bpuffy\s*eyes\b", r"\beye\s*bags\b", r"\bswollen\s*eyes\b"
    ],
    "Stretch Marks": [
        r"\bstretch\s*marks?\b", r"\bstriae\b", r"\bbio\s*oil\b"
    ],
    "Nerve Health": [
        r"\bnerve(s)?\b", r"\bnumb(ness)?\b", r"\btingling\b", r"\bneuropathy\b"
    ],
    "Oral Care": [
        r"\boral\b", r"\bmouth(wash)?\b", r"\bgums?\b", r"\bbad\s*breath\b",
        r"\bhalitosis\b"
    ],
    "Dental Health": [
        r"\bdental\b", r"\bteeth\b", r"\btooth(ache)?\b", r"\bcavit(y|ies)\b"
    ],
    "Nail Health": [
        r"\bnail(s)?\b", r"\bbrittle\s*nails\b", r"\bcuticles?\b"
    ],
    "Motion Sickness": [
        r"\bmotion\s*sickness\b", r"\bsea\s*sickness\b", r"\bnausea\b",
        r"\bvomiting\b", r"\bcar\s*sick(ness)?\b"
    ],
    "Smokers": [
        r"\bsmoker(s)?\b", r"\bsmoking\b", r"\bnicotine\b", r"\bquit\s*smoking\b"
    ],

}