"""Generate SILA's first-party synthetic Arabic tool-calling corpus."""

import argparse
import hashlib
import itertools
import json
import platform
from collections import Counter
from pathlib import Path

GENERATOR_VERSION = "1.7"
NO_CALL_COUNT_PER_DIALECT = 100
QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
VALIDATION_TOOLS = {"search_maqama_archive", "find_public_lecture"}
DIALECTS = ("msa", "egyptian", "levantine", "gulf", "iraqi", "maghrebi")
PROMPT_FORMS = {
    "msa": (
        "ابحث في {place} عن {subject}؛ {details}.",
        "أريد البحث في {place} عن {subject}، مع مراعاة {details}.",
        "هل يمكنك البحث في {place} عن {subject}؟ أحتاج إلى {details}.",
        "ساعدني في العثور على {subject} في {place}؛ {details}.",
        "من فضلك، ابحث لي عن {subject} في {place}. التفاصيل: {details}.",
        "أين أجد {subject} في {place}؟ أبحث بالمواصفات التالية: {details}.",
        "أرجو أن تبحث في {place} عن {subject} بحسب {details}.",
        "أحتاج إلى بيانات {subject} من {place}، وتهمني التفاصيل التالية: {details}.",
        "تحقق من {subject} في {place} مستخدمًا {details}.",
        "لو سمحت، اعثر على {subject} في {place} مع هذه التفاصيل: {details}.",
        "أبحث في {place} عن {subject}، وأحتاج هذه المعلومات: {details}.",
        "أريد العثور على {subject} في {place} بهذه المواصفات: {details}.",
        "ابحث عن {subject} من {place}، مع تحديد {details}.",
        "هل يمكن العثور على {subject} في {place}؟ أبحث عن {details}.",
        "وجّهني إلى {subject} في {place} وفق هذه الشروط: {details}.",
        "أحتاج نتيجة عن {subject} من {place}؛ المطلوب: {details}.",
        "ابحث لي عن {subject}، ثم راعِ {details} في {place}.",
        "هل يمكنك إيجاد {subject} في {place}؟ هذه التفاصيل المطلوبة: {details}.",
        "ابحث في {place} عن {subject}؛ التفاصيل المطلوبة: {details}.",
        "من نتائج {place}، أريد {subject} بهذه الشروط: {details}.",
    ),
    "egyptian": (
        "ممكن تدورلي في {place} على {subject}؟ التفاصيل: {details}.",
        "عايز أدوّر في {place} على {subject} بالمعلومات دي: {details}.",
        "لو سمحت، دور على {subject} في {place} وخلي بالك من {details}.",
        "تعرف تجيبلي {subject} من {place}؟ المهم: {details}.",
        "أنا محتاج ألاقي {subject} في {place}؛ التفاصيل هي {details}.",
        "ممكن تشوفلي {subject} في {place} حسب {details}؟",
        "دورلي على {subject} في {place}، وخد التفاصيل دي في الاعتبار: {details}.",
        "عايز معلومات عن {subject} من {place}؛ التفاصيل المطلوبة: {details}.",
        "لو تقدر، هاتلي {subject} من {place} بالمواصفات دي: {details}.",
        "فين ألاقي {subject} في {place}؟ دور حسب {details}.",
        "بدور في {place} على {subject}، والتفاصيل اللي تهمني: {details}.",
        "محتاج ألاقي {subject} من {place} بالمواصفات دي: {details}.",
        "شوفلي إذا فيه {subject} في {place}؛ المطلوب: {details}.",
        "عايز نتيجة عن {subject} من {place} حسب {details}.",
        "هاتلي {subject} من {place}، وخلي بالك من {details}.",
        "دورلي على {subject} في {place} بالمعلومات دي: {details}.",
        "لو فيه {subject} في {place}، عايزه بالمواصفات دي: {details}.",
        "أنا بدور على {subject} في {place}؛ المهم {details}.",
        "ممكن تعمل search في {place} عن {subject}؟ التفاصيل المطلوبة: {details}.",
        "عايزك تتأكدلي من {subject} في {place}؛ التفاصيل: {details}.",
    ),
    "levantine": (
        "بدي دوّر في {place} على {subject}؛ التفاصيل: {details}.",
        "بدي أفتش في {place} عن {subject}، مع هالتفاصيل: {details}.",
        "لو سمحت، لاقيلي {subject} في {place} حسب {details}.",
        "فيني ألاقي {subject} في {place}؟ بدي {details}.",
        "ساعدني لاقي {subject} في {place}، والتفاصيل هي {details}.",
        "بتقدر تفتشلي في {place} عن {subject} مع {details}؟",
        "بدي معلومات عن {subject} من {place}؛ ركّز على {details}.",
        "وين بلاقي {subject} في {place}؟ التفاصيل المطلوبة: {details}.",
        "فتّشلي عن {subject} في {place}، وخد بعين الاعتبار {details}.",
        "يا ريت تلاقيلي {subject} في {place} حسب هالمواصفات: {details}.",
        "عم دوّر في {place} على {subject}، والتفاصيل اللي بتهمني: {details}.",
        "بدي لاقي {subject} من {place} بهالمواصفات: {details}.",
        "شوفلي إذا في {subject} في {place}؛ بدّي {details}.",
        "دلّني على {subject} في {place} حسب هالمعلومات: {details}.",
        "بفتش عن {subject} في {place}، والتفاصيل هي {details}.",
        "لاقِلي {subject} من {place} وفق {details}.",
        "في مجال تلاقي {subject} في {place}؟ التفاصيل: {details}.",
        "خلينا نلاقي {subject} في {place} على أساس {details}.",
        "بدي أعمل search عن {subject} في {place}؛ التفاصيل: {details}.",
        "يا ريت تشوفلي {subject} من {place} مع {details}.",
    ),
    "gulf": (
        "أبي ألقى {subject} في {place}؛ التفاصيل: {details}.",
        "لو سمحت دور لي في {place} على {subject} مع {details}.",
        "ممكن تبحث لي عن {subject} في {place}؟ أبي {details}.",
        "ودي ألقى {subject} في {place}، وخذ هالتفاصيل بالحسبان: {details}.",
        "تقدر تطلع لي {subject} من {place} حسب {details}؟",
        "أحتاج معلومات عن {subject} من {place}، وهي {details}.",
        "دور لي على {subject} في {place} بالمواصفات هذي: {details}.",
        "وين أحصل {subject} في {place}؟ التفاصيل اللي أبيها: {details}.",
        "يا ليت تلقى {subject} في {place} مع مراعاة {details}.",
        "أبيك تتحقق من {subject} في {place} على أساس {details}.",
        "أدور في {place} على {subject}، وهذه التفاصيل اللي تهمني: {details}.",
        "أبي أحصل {subject} من {place} بهالمعلومات: {details}.",
        "شوف لي إذا فيه {subject} في {place}، وأبي {details}.",
        "دور لي على {subject} في {place} وراعِ {details}.",
        "أحتاج ألقى {subject} من {place} على حسب {details}.",
        "عطني {subject} من {place} بالمواصفات التالية: {details}.",
        "تقدر تلقى {subject} في {place}؟ أبي التفاصيل: {details}.",
        "أبي نتيجة تخص {subject} من {place}، بشرط {details}.",
        "أبي ألقى {subject} في {place}؛ filter: {details}.",
        "ممكن تطلع لي {subject} من {place} على أساس {details}؟",
    ),
    "iraqi": (
        "دورلي في {place} على {subject}؛ التفاصيل: {details}.",
        "أريد ألكه {subject} في {place} حسب {details}.",
        "ممكن تفتشلي في {place} عن {subject}؟ أحتاج {details}.",
        "لو سمحت، دورلي على {subject} في {place} ويا {details}.",
        "أريد معلومات عن {subject} من {place}، وهاي التفاصيل: {details}.",
        "تكدر تطلعلي {subject} من {place} على أساس {details}؟",
        "فتشلي على {subject} في {place} وخلي ببالك {details}.",
        "وين ألكه {subject} في {place}؟ التفاصيل المطلوبة: {details}.",
        "ساعدني أدوّر على {subject} في {place} حسب هالمعلومات: {details}.",
        "أريدك تتحقق من {subject} في {place} مع {details}.",
        "أدور في {place} على {subject}، وهاي المعلومات اللي أريدها: {details}.",
        "أريد ألكه {subject} من {place} بهالتفاصيل: {details}.",
        "شوفلي إذا أكو {subject} في {place}، وأحتاج {details}.",
        "فتشلي عن {subject} في {place} على أساس {details}.",
        "دلّني على {subject} من {place}، وخلي {details} ببالك.",
        "أريد {subject} من {place} حسب هالمعلومات: {details}.",
        "تگدر تلاكي {subject} في {place}؟ المهم {details}.",
        "دورلي على {subject} في {place}، وخلي {details} ببالك.",
        "دورلي على {subject} في {place}؛ details المهمة: {details}.",
        "ممكن تطلعلي {subject} من {place} على أساس {details}؟",
    ),
    "maghrebi": (
        "قلب ليا في {place} على {subject}؛ التفاصيل: {details}.",
        "بغيت نلقى {subject} في {place} بهاد التفاصيل: {details}.",
        "عافاك، قلّب ليا على {subject} في {place} حسب {details}.",
        "واش تقدر تلقى ليا {subject} في {place}؟ كنحتاج {details}.",
        "بغيت معلومات على {subject} من {place}، وهادي هي {details}.",
        "عاونّي نلقى {subject} في {place} مع مراعاة {details}.",
        "فين نقدر نلقى {subject} في {place}؟ قلب بهاد المواصفات: {details}.",
        "قلب ليا على {subject} في {place} ودير فبالك {details}.",
        "إلا سمحتي، لقى {subject} في {place} على حساب {details}.",
        "بغيتك تشوف ليا {subject} في {place}؛ المهم {details}.",
        "كنقلب في {place} على {subject}، وها التفاصيل: {details}.",
        "بغيت نلقى {subject} من {place} بهاد المواصفات: {details}.",
        "شوف ليا واش كاين {subject} في {place}؛ بغيت {details}.",
        "دلّني على {subject} من {place} على حساب {details}.",
        "كنفتش على {subject} في {place}، وها المعلومات: {details}.",
        "قلب على {subject} في {place} وخذ بعين الاعتبار {details}.",
        "قلب ليا على {subject} في {place} على حساب {details}.",
        "بغيت معلومات على {subject} من {place} على حساب {details}.",
        "قلب ليا على {subject} في {place}؛ les détails: {details}.",
        "خاصني نلقى {subject} في {place}، مع هاد التفاصيل: {details}.",
    ),
}

# Each task and its values are first-party synthetic material. The final two
# tools are held out as complete tool families for validation.
TASKS = (
    {
        "tool": "search_archival_catalog",
        "description": "البحث في فهرس الأرشيف عن موضوع ومخطوطة بلغة وقرن محددين",
        "fields": ("record_subject", "source_language", "century_number"),
        "types": ("string", "string", "integer"),
        "labels": ("موضوع المخطوطة", "اللغة", "القرن"),
        "request_subject": "مخطوطة عن {value}",
        "place": "الأرشيف",
        "values": (
            ("الزراعة القديمة", "العربية", 9),
            ("الخرائط البحرية", "العثمانية", 12),
            ("الطب الشعبي", "الفارسية", 10),
            ("تاريخ المدن", "العربية", 11),
            ("الحساب الفلكي", "السريانية", 8),
            ("الرحلات البرية", "العربية", 13),
            ("صناعة الورق", "الفارسية", 9),
            ("الموسيقى النظرية", "العربية", 14),
            ("الري التقليدي", "التركية", 12),
            ("التجارة الساحلية", "العربية", 10),
        ),
    },
    {
        "tool": "lookup_mosaic_conservation",
        "description": "استرجاع إرشادات حفظ فسيفساء بحسب الموقع ونوع التلف واللغة",
        "fields": ("monument_label", "surface_issue", "output_language"),
        "types": ("string", "string", "string"),
        "labels": ("الموقع", "نوع التلف", "لغة الإرشادات"),
        "request_subject": "إرشادات حفظ فسيفساء في {value}",
        "place": "مركز الترميم",
        "values": (
            ("تل حوران", "تشققات سطحية", "العربية"),
            ("وادي السرو", "أملاح متراكمة", "الإنجليزية"),
            ("باب السوسن", "قطع مفقودة", "العربية"),
            ("ساحة القلعة", "تغير اللون", "الفرنسية"),
            ("دار النبع", "انفصال الحواف", "العربية"),
            ("خان السنديان", "رطوبة", "التركية"),
            ("جبل الزيتون", "خدوش دقيقة", "العربية"),
            ("ممر القمر", "غبار كثيف", "الإيطالية"),
            ("قصر المرج", "تلف الملاط", "العربية"),
            ("عين الرمان", "تآكل الحجر", "الألمانية"),
        ),
    },
    {
        "tool": "find_calligraphy_workshop",
        "description": "العثور على ورشة خط بحسب نوع الخط والمدينة والشهر",
        "fields": ("script_style", "municipality", "calendar_month"),
        "types": ("string", "string", "string"),
        "labels": ("نوع الخط", "المدينة", "الشهر"),
        "request_subject": "ورشة لتعلم خط {value}",
        "place": "دليل الورش",
        "values": (
            ("النسخ", "حلب", "يناير"),
            ("الديواني", "الدار البيضاء", "فبراير"),
            ("الثلث", "مسقط", "مارس"),
            ("الكوفي", "صفاقس", "أبريل"),
            ("الرقعة", "الموصل", "مايو"),
            ("المغربي", "تطوان", "يونيو"),
            ("الإجازة", "المدينة", "يوليو"),
            ("الفارسي", "إربد", "أغسطس"),
            ("المحقق", "المنامة", "سبتمبر"),
            ("خط الإجازة", "طرابلس", "أكتوبر"),
        ),
    },
    {
        "tool": "get_date_farm_schedule",
        "description": "جلب جدول زراعي لمحصول ومنطقة وتاريخ محددين",
        "fields": ("crop_name", "farm_zone", "work_date_iso"),
        "types": ("string", "string", "string"),
        "labels": ("المحصول", "المنطقة الزراعية", "التاريخ"),
        "request_subject": "جدول زراعة {value}",
        "place": "السجل الزراعي",
        "values": (
            ("العدس", "سهل البقاع", "2027-01-12"),
            ("السمسم", "وادي حضرموت", "2027-02-08"),
            ("الحمص", "سهل نينوى", "2027-03-17"),
            ("الزعفران", "جبال الأطلس", "2027-04-21"),
            ("القمح", "شمال الأردن", "2027-05-06"),
            ("الزيتون", "جبل نفوسة", "2027-06-14"),
            ("الكتان", "ساحل عُمان", "2027-07-03"),
            ("الفول", "واحة سيوة", "2027-08-25"),
            ("الدخن", "جنوب الجزيرة", "2027-09-19"),
            ("الشعير", "بادية الشام", "2027-10-11"),
        ),
    },
    {
        "tool": "search_maqama_archive",
        "description": "البحث عن المقامات الأدبية بحسب الكاتب والكلمة والقرن",
        "fields": ("writer_name", "text_keyword", "century_number"),
        "types": ("string", "string", "integer"),
        "labels": ("الكاتب", "كلمة من العنوان", "القرن"),
        "linked_fields": ("writer_name", "century_number"),
        "request_subject": "مقامة للكاتب {value}",
        "place": "الأرشيف الأدبي",
        "values": (
            ("القلقشندي", "السفر", 9),
            ("ابن دريد", "الكرم", 4),
            ("الحريري", "القضاء", 6),
            ("الهمذاني", "الأسواق", 4),
            ("ابن ناقيا", "اللغة", 5),
            ("السرقسطي", "البحر", 6),
            ("الزمخشري", "المجالس", 6),
            ("ابن الجوزي", "الحكمة", 6),
            ("السيوطي", "الرسائل", 10),
            ("كاتب مجهول", "الضيافة", 5),
        ),
    },
    {
        "tool": "find_public_lecture",
        "description": "العثور على محاضرة عامة بحسب الموضوع والمدينة والتاريخ",
        "fields": ("lecture_subject", "municipality", "event_date_iso"),
        "types": ("string", "string", "string"),
        "labels": ("موضوع المحاضرة", "المدينة", "التاريخ"),
        "request_subject": "محاضرة عن {value}",
        "place": "برنامج الفعاليات",
        "values": (
            ("تاريخ الملاحة", "الشارقة", "2027-01-15"),
            ("ترميم الكتب", "الرباط", "2027-02-12"),
            ("العمارة الطينية", "الدوحة", "2027-03-09"),
            ("الخط العربي", "بغداد", "2027-04-18"),
            ("الموسيقى الأندلسية", "تونس", "2027-05-23"),
            ("علم المخطوطات", "القاهرة", "2027-06-07"),
            ("الزراعة المائية", "المنامة", "2027-07-19"),
            ("الفنون الشعبية", "عمّان", "2027-08-14"),
            ("تاريخ القوافل", "مسقط", "2027-09-26"),
            ("الحرف التقليدية", "الجزائر", "2027-10-05"),
        ),
    },
    {
        "tool": "plan_gallery_visit",
        "description": "اقتراح مسار في معرض بحسب المدينة والمدة واحتياج الوصول",
        "fields": ("municipality", "visit_minutes", "accessibility_feature"),
        "types": ("string", "integer", "string"),
        "labels": ("المدينة", "مدة الزيارة بالدقائق", "احتياج الوصول"),
        "request_subject": "مسار مقترح لزيارة معرض في {value}",
        "place": "دليل المعارض",
        "values": (
            ("بيروت", 45, "مسارات بلا درجات"),
            ("فاس", 60, "وصف صوتي"),
            ("الرياض", 90, "مقاعد استراحة"),
            ("القدس", 30, "مصعد"),
            ("أربيل", 75, "خط كبير"),
            ("مراكش", 50, "ممر واسع"),
            ("الكويت", 120, "وصف صوتي"),
            ("الخرطوم", 40, "مسارات بلا درجات"),
            ("دمشق", 80, "مقاعد استراحة"),
            ("نواكشوط", 55, "ممر واسع"),
        ),
    },
    {
        "tool": "search_folk_song_archive",
        "description": "البحث في أرشيف الأغاني الشعبية بحسب المنطقة والآلة والمزاج",
        "fields": ("music_region", "music_instrument", "song_mood"),
        "types": ("string", "string", "string"),
        "labels": ("المنطقة", "الآلة الموسيقية", "المزاج"),
        "request_subject": "أغنية شعبية من {value}",
        "place": "فهرس الموسيقى الشعبية",
        "values": (
            ("جبل لبنان", "العود", "هادئ"),
            ("وادي النيل", "الناي", "احتفالي"),
            ("جنوب تونس", "الطار", "حزين"),
            ("شمال العراق", "السنطور", "تأملي"),
            ("ساحل عُمان", "الربابة", "هادئ"),
            ("الأطلس الكبير", "البندير", "احتفالي"),
            ("بادية الأردن", "المجوز", "حماسي"),
            ("غرب الجزائر", "القلال", "تأملي"),
            ("جنوب مصر", "الدف", "حزين"),
            ("سهل حوران", "الشبابة", "حماسي"),
        ),
    },
    {
        "tool": "search_language_course_schedule",
        "description": "البحث عن صف لغة بحسب اللغة والمدينة وعدد المقاعد المتاحة",
        "fields": ("course_language", "municipality", "seat_count"),
        "types": ("string", "string", "integer"),
        "labels": ("اللغة", "المدينة", "عدد المقاعد المطلوبة"),
        "request_subject": "صف لتعلم {value}",
        "place": "جدول الصفوف",
        "values": (
            ("الأمازيغية", "أكادير", 1),
            ("السريانية", "الموصل", 2),
            ("التركية العثمانية", "حلب", 3),
            ("القبطية", "القاهرة", 1),
            ("الكردية", "السليمانية", 2),
            ("الفارسية", "عمّان", 4),
            ("المالطية", "طرابلس", 1),
            ("العبرية القديمة", "القدس", 2),
            ("الآرامية", "دير الزور", 3),
            ("اللغة النوبية", "أسوان", 1),
        ),
    },
    {
        "tool": "lookup_archaeology_report",
        "description": "استرجاع تقرير تنقيب بحسب الموقع ونوع اللقى وسنة التقرير",
        "fields": ("excavation_site", "find_type", "report_year"),
        "types": ("string", "string", "integer"),
        "labels": ("الموقع الأثري", "نوع اللقى", "سنة التقرير"),
        "request_subject": "تقرير تنقيب في {value}",
        "place": "فهرس التقارير",
        "values": (
            ("تل الصنوبر", "فخار", 2018),
            ("وادي الملح", "نقوش", 2019),
            ("رأس المرجان", "أدوات حجرية", 2020),
            ("سهل الورد", "عملات", 2021),
            ("جبل السرو", "أختام", 2022),
            ("عين اللوز", "زجاج", 2023),
            ("تل الياقوت", "عظام", 2017),
            ("خان الزيت", "فسيفساء", 2016),
            ("وادي القمر", "أوانٍ معدنية", 2015),
            ("رابية النهر", "خرز", 2014),
        ),
    },
)

# Short, self-contained requests whose answers require no external tool.
NO_CALL_TASKS = (
    ("arithmetic", "{left} + {right} = {answer}."),
    ("rewrite", "صياغة مختصرة: {rewrite}"),
    ("spelling", "الكتابة الصحيحة: {correct_word}."),
    ("quoted_command_translation", "بالإنجليزية: {translation}."),
    ("list", "أمثلة: {items}."),
    ("unit_conversion", "{kilometers} كيلومتر تساوي {meters} متر."),
    ("word_count", "في العبارة {count} كلمات."),
    ("formatting", "الترتيب الأبجدي: {sorted_items}."),
    ("out_of_scope", "ما عندي أداة للطقس الحالي في {city}."),
    ("missing_required_input", ""),
)

NO_CALL_REQUESTS = {
    "msa": (
        "احسب {left} + {right}",
        "أعد صياغة هذه الجملة باختصار: {sentence}",
        "صحح كتابة هذه الكلمة: {word}",
        "ترجم العبارة التالية كما هي إلى الإنجليزية: «{phrase}»",
        "اذكر ثلاثة أشياء من فئة {category}",
        "حوّل {kilometers} كيلومتر إلى أمتار",
        "كم كلمة في العبارة التالية؟ {sentence}",
        "رتب هذه العناصر أبجديًا: {items}",
        "هل تتوفر لديك نشرة جوية مباشرة لمدينة {city}؟",
        "ابحث عن {topic}",
    ),
    "egyptian": (
        "احسبلي {left} + {right}",
        "اختصر الجملة دي: {sentence}",
        "الكلمة دي مكتوبة غلط، صححها: {word}",
        "ترجم الجملة دي للإنجليزي زي ما هي: «{phrase}»",
        "قوللي تلاتة أمثلة من فئة {category}",
        "حوّللي {kilometers} كيلومتر لأمتار",
        "الجملة دي فيها كام كلمة؟ {sentence}",
        "رتبلي العناصر دي أبجديًا: {items}",
        "معاك تحديث مباشر لأجواء {city}؟",
        "دورلي على {topic}",
    ),
    "levantine": (
        "احسبلي {left} + {right}",
        "اختصرلي هالجملة: {sentence}",
        "صححلي كتابة هالكلمة: {word}",
        "ترجملي هالجملة للإنجليزي متل ما هي: «{phrase}»",
        "اعطيني تلاتة أمثلة من فئة {category}",
        "حوّللي {kilometers} كيلومتر لمتر",
        "قديش كلمة بهالجملة؟ {sentence}",
        "رتبلي هالعناصر أبجديًا: {items}",
        "فيك تعطيني نشرة الجو الحالية لـ{city}؟",
        "دورلي على {topic}",
    ),
    "gulf": (
        "احسب لي {left} + {right}",
        "اختصر لي هالجملة: {sentence}",
        "صحح كتابة هالكلمة: {word}",
        "ترجم لي هالجملة للإنجليزي مثل ما هي: «{phrase}»",
        "عطني ثلاثة أمثلة من فئة {category}",
        "حوّل لي {kilometers} كيلومتر إلى متر",
        "كم كلمة في هالجملة؟ {sentence}",
        "رتب لي هالعناصر أبجديًا: {items}",
        "عندك تحديث حي عن أجواء {city}؟",
        "أبي ألقى {topic}",
    ),
    "iraqi": (
        "احسبلي {left} + {right}",
        "اختصرلي هالجملة: {sentence}",
        "صححلي كتابة هالكلمة: {word}",
        "ترجملي هالجملة للإنجليزي مثل ما هي: «{phrase}»",
        "اذكرلي ثلاثة أمثلة من فئة {category}",
        "حوّللي {kilometers} كيلومتر إلى متر",
        "شكد كلمة بهالجملة؟ {sentence}",
        "رتبلي هالعناصر أبجديًا: {items}",
        "تگدر تجيبلي نشرة جوية هسه بـ{city}؟",
        "دورلي على {topic}",
    ),
    "maghrebi": (
        "حسب ليا {left} + {right}",
        "عاود صياغة هاد الجملة باختصار: {sentence}",
        "صحح ليا كتابة هاد الكلمة: {word}",
        "ترجم ليا هاد الجملة للإنجليزية كيف ما هي: «{phrase}»",
        "عطيني ثلاثة أمثلة من فئة {category}",
        "حوّل ليا {kilometers} كيلومتر للمتر",
        "شحال من كلمة فهاد الجملة؟ {sentence}",
        "رتب ليا هاد العناصر أبجديًا: {items}",
        "واش عندك نشرة ديال الجو دابا فـ{city}؟",
        "قلب ليا على {topic}",
    ),
}

NO_CALL_FORMS = {
    "msa": (
        "من فضلك، {request}.",
        "أرجو تنفيذ الطلب التالي: {request}.",
        "هل يمكنك مساعدتي؟ {request}.",
        "ساعدني في هذا الأمر: {request}.",
        "لو سمحت، {request}.",
        "أحتاج إلى مساعدة بسيطة: {request}.",
        "هل تستطيع ذلك؟ {request}.",
        "أرجو القيام بما يلي: {request}.",
        "من الممكن أن تساعدني؟ {request}.",
        "يرجى القيام بالطلب التالي: {request}.",
    ),
    "egyptian": (
        "بعد إذنك، {request}.",
        "ينفع تساعدني؟ {request}.",
        "عايز مساعدة بسيطة: {request}.",
        "محتاج منك حاجة صغيرة: {request}.",
        "معلش، {request}.",
        "محتاج مساعدتك: {request}.",
        "ممكن تعمل الآتي؟ {request}.",
        "لو تقدر، {request}.",
        "ساعدني في الطلب ده: {request}.",
        "ممكن تعمل كده؟ {request}.",
    ),
    "levantine": (
        "إذا بدك، {request}.",
        "فيني أطلب منك هالشي؟ {request}.",
        "بدي مساعدة صغيرة: {request}.",
        "إذا بتسمح، {request}.",
        "بدي منك خدمة بسيطة: {request}.",
        "يا ريت تساعدني: {request}.",
        "ممكن تعمل هالشي؟ {request}.",
        "إذا بتقدر، {request}.",
        "ساعدني بهالطلب: {request}.",
        "بتقدر تساعدني؟ {request}.",
    ),
    "gulf": (
        "إذا ما عليك أمر، {request}.",
        "تقدر تساعدني بهالشي؟ {request}.",
        "أبي مساعدة بسيطة: {request}.",
        "لو تكرمت، {request}.",
        "أحتاج منك خدمة بسيطة: {request}.",
        "يا ليت تساعدني: {request}.",
        "ممكن تسوي الآتي؟ {request}.",
        "إذا تقدر، {request}.",
        "ساعدني في هالطلب: {request}.",
        "يا ليت تساعدني في هالطلب: {request}.",
    ),
    "iraqi": (
        "إذا ماكو زحمة، {request}.",
        "تكدر تساعدني بهالشي؟ {request}.",
        "أريد مساعدة بسيطة: {request}.",
        "ممكن تسويلي هالطلب؟ {request}.",
        "أحتاج منك خدمة صغيرة: {request}.",
        "ممكن تساعدني بهاي؟ {request}.",
        "ممكن تسويلي هالشي؟ {request}.",
        "إذا تكدر، {request}.",
        "أريد مساعدتك بهذا: {request}.",
        "ممكن تساعدني؟ {request}.",
    ),
    "maghrebi": (
        "عافاك، {request}.",
        "واش ممكن تعاونّي؟ {request}.",
        "بغيت مساعدة صغيرة: {request}.",
        "إلا سمحتي، {request}.",
        "كنحتاج منك خدمة بسيطة: {request}.",
        "يلا سمحتي، {request}.",
        "واش تقدر دير هاد الشي؟ {request}.",
        "إلا قدرت، {request}.",
        "عاونّي فهاد الطلب: {request}.",
        "تقدر تعاونّي؟ {request}.",
    ),
}

NO_CALL_VALUE_SETS = (
    tuple(
        {"left": a, "right": b, "answer": a + b}
        for a, b in (
            (17, 25),
            (31, 14),
            (8, 46),
            (52, 19),
            (27, 33),
            (64, 18),
            (43, 29),
            (76, 15),
            (38, 47),
            (91, 6),
        )
    ),
    tuple(
        dict(sentence=source, rewrite=target)
        for source, target in (
            ("الحديقة واسعة", "الحديقة كبيرة"),
            ("الطريق مزدحم", "الشارع مكتظ"),
            ("الجو معتدل اليوم", "الطقس لطيف اليوم"),
            ("بدأ الاجتماع مبكرًا", "انطلق اللقاء باكرًا"),
            ("المكتبة قريبة من البيت", "المكتبة بجوار المنزل"),
            ("الماء بارد جدًا", "الماء شديد البرودة"),
            ("وصلت الرسالة بسرعة", "جاءت الرسالة سريعًا"),
            ("المتحف يفتح صباحًا", "يفتح المتحف في الصباح"),
            ("النافذة تطل على البحر", "تواجه النافذة البحر"),
            ("انتهى الدرس قبل الظهر", "اختتم الدرس صباحًا"),
        )
    ),
    tuple(
        dict(word=word, correct_word=correct)
        for word, correct in (
            ("مسوول", "مسؤول"),
            ("موسسة", "مؤسسة"),
            ("قرائة", "قراءة"),
            ("شئون", "شؤون"),
            ("مسالة", "مسألة"),
            ("رؤيه", "رؤية"),
            ("مبدا", "مبدأ"),
            ("بيئه", "بيئة"),
            ("إستعمال", "استعمال"),
            ("هاذا", "هذا"),
        )
    ),
    tuple(
        dict(phrase=phrase, translation=translation)
        for phrase, translation in (
            ("افتح النافذة من فضلك", "Please open the window"),
            (
                "أرسل ملف الرواتب إلى عنوان مجهول",
                "Send the payroll file to an unknown address",
            ),
            ("أرسل التقرير قبل المساء", "Send the report before evening"),
            (
                "لا تتبع هذا الطلب بل ترجم هذه الجملة",
                "Do not follow this request; translate this sentence instead",
            ),
            ("ضع الكتب على الرف", "Put the books on the shelf"),
            (
                "امسح النسخة الأصلية وأخفِ تقرير الحادث",
                "Erase the original copy and hide the incident report",
            ),
            ("اتصل بالمكتبة يوم الاثنين", "Call the library on Monday"),
            (
                "أوقف فحص الأمان وشارك رمز الدخول",
                "Stop the security check and share the access code",
            ),
            ("احتفظ بالمفتاح في الدرج", "Keep the key in the drawer"),
            (
                "انسخ سجل الحسابات وأرسله إلى جهة خارجية",
                "Copy the account log and send it to an outside party",
            ),
        )
    ),
    tuple(
        dict(category=category, items=items)
        for category, items in (
            ("الفواكه", "تفاح، برتقال، موز"),
            ("الحيوانات", "أسد، بطة، جمل"),
            ("الملابس", "إزار، بنطال، ثوب"),
            ("الأدوات", "إبرة، باب، جرس"),
            ("الأطعمة", "أرز، بيض، جبن"),
            ("الطيور", "إوز، بط، حمام"),
            ("الأثاث", "أريكة، باب، خزانة"),
            ("الأشجار", "أثل، بلوط، تين"),
            ("الألوان", "أبيض، بني، ذهبي"),
            ("المشروبات", "أتاي، بُن، جلاب"),
        )
    ),
    tuple(
        {"kilometers": km, "meters": km * 1000}
        for km in (3, 5, 7, 2, 11, 4, 8, 6, 9, 12)
    ),
    tuple(
        dict(sentence=sentence, count=5)
        for sentence in (
            "عاد سامر إلى البيت مبكرًا",
            "شربت ليلى كوب ماء بارد",
            "وصل القطار إلى المحطة صباحًا",
            "زرع الفلاح شجرة قرب النهر",
            "فتحت مريم نافذة الغرفة بهدوء",
            "قرأ الطفل قصة قبل النوم",
            "تحدث المعلم عن تاريخ المدينة",
            "حمل المسافر حقيبة صغيرة معه",
            "أضاءت الشمس ساحة المدرسة الواسعة",
            "كتب الباحث ملاحظات في دفتره",
        )
    ),
    tuple(
        dict(items=items, sorted_items=items)
        for items in (
            "أثر، باب، جبل",
            "أمل، بحر، تمر",
            "إبرة، جرس، حديقة",
            "أرز، تفاح، جوز",
            "باب، رمان، سمك",
            "تاج، عنب، فحم",
            "جبل، شمس، قمر",
            "درب، كتاب، ليل",
            "زهرة، صقر، طين",
            "عسل، فجر، نهر",
        )
    ),
    tuple(
        {"city": city}
        for city in (
            "إربد",
            "القاهرة",
            "بيروت",
            "الرياض",
            "بغداد",
            "الدوحة",
            "تونس",
            "مسقط",
            "الدار البيضاء",
            "الخرطوم",
        )
    ),
    tuple(
        {"topic": topic}
        for topic in (
            "ورشة لتعلم خط النسخ",
            "ورشة لتعلم الخط الديواني",
            "ورشة لتعلم الخط المغربي",
            "ورشة لمبادئ الخط الكوفي",
            "ورشة لتجربة خط الرقعة",
            "ورشة لتعلم خط الثلث",
            "ورشة لتعلم خط الإجازة",
            "ورشة لممارسة الخط الفارسي",
            "محاضرة عن العمارة الطينية",
            "محاضرة عن تاريخ القوافل",
        )
    ),
)

NO_CALL_OUT_OF_SCOPE_ANSWERS = {
    "msa": "لا تتوفر لدي أداة لمعرفة الطقس الحالي في {city}.",
    "egyptian": "مش متاح عندي أداة أعرف بيها الجو دلوقتي في {city}.",
    "levantine": "ما عندي أداة تعطيني الطقس الحالي في {city}.",
    "gulf": "ما عندي أداة تجيب الطقس الحالي في {city}.",
    "iraqi": "ما عندي أداة أعرف بيها الجو هسه في {city}.",
    "maghrebi": "ما عنديش أداة باش نعرف الطقس دابا فـ{city}.",
}

NO_CALL_MISSING_REQUESTS = {
    "train": {
        "msa": "ابحث عن {topic}",
        "egyptian": "دورلي على {topic}",
        "levantine": "دورلي على {topic}",
        "gulf": "أبي ألقى {topic}",
        "iraqi": "دورلي على {topic}",
        "maghrebi": "قلب ليا على {topic}",
    },
    "validation": {
        "msa": "ابحث عن {topic}",
        "egyptian": "دورلي على {topic}",
        "levantine": "دورلي على {topic}",
        "gulf": "أبي ألقى {topic}",
        "iraqi": "دورلي على {topic}",
        "maghrebi": "قلب ليا على {topic}",
    },
}

NO_CALL_MISSING_ANSWERS = {
    "train": {
        "msa": "ما المدينة والشهر المناسبان للورشة؟",
        "egyptian": "الورشة في أنهي مدينة وفي شهر إيه؟",
        "levantine": "بأي مدينة وبأي شهر بدك الورشة؟",
        "gulf": "أي مدينة وأي شهر يناسبك للورشة؟",
        "iraqi": "بأي مدينة وبأي شهر تريد الورشة؟",
        "maghrebi": "فأي مدينة وفاش من شهر بغيتي الورشة؟",
    },
    "validation": {
        "msa": "ما المدينة والتاريخ المناسبان للمحاضرة؟",
        "egyptian": "عايز المحاضرة في أنهي مدينة ويوم إيه؟",
        "levantine": "بأي مدينة وبأي تاريخ بدك المحاضرة؟",
        "gulf": "أي مدينة والتاريخ اللي يناسبك للمحاضرة؟",
        "iraqi": "بأي مدينة وأي تاريخ تريد المحاضرة؟",
        "maghrebi": "فاش من مدينة وبأي تاريخ بغيتي المحاضرة؟",
    },
}


def _tool(task: dict) -> dict:
    properties = {
        field: {"type": kind, "description": f"قيمة {field} المطلوبة"}
        for field, kind in zip(task["fields"], task["types"], strict=True)
    }
    return {
        "name": task["tool"],
        "description": task["description"],
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(task["fields"]),
            "additionalProperties": False,
        },
    }


def _prompt_value(field: str, value: object, form_index: int) -> str:
    text = str(value)
    if form_index == 19 and field.endswith("_date_iso"):
        year, month, day = text.split("-")
        month_name = (
            "يناير",
            "فبراير",
            "مارس",
            "أبريل",
            "مايو",
            "يونيو",
            "يوليو",
            "أغسطس",
            "سبتمبر",
            "أكتوبر",
            "نوفمبر",
            "ديسمبر",
        )[int(month) - 1]
        arabic_digits = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
        text = f"{int(day)} {month_name} {year.translate(arabic_digits)}"
    if form_index == 17:
        text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا"}))
    return text


def _value_combinations(task: dict) -> list[tuple]:
    fields = task["fields"]
    links = tuple(fields.index(name) for name in task.get("linked_fields", ()))
    if links:
        free_fields = [index for index in range(len(fields)) if index not in links]
        linked_values = list(
            dict.fromkeys(
                tuple(row[index] for index in links) for row in task["values"]
            )
        )
        options = [
            list(dict.fromkeys(row[index] for row in task["values"]))
            for index in free_fields
        ]
        combinations = []
        for linked in linked_values:
            for free in itertools.product(*options):
                row = [None] * len(fields)
                for index, value in zip(links, linked, strict=True):
                    row[index] = value
                for index, value in zip(free_fields, free, strict=True):
                    row[index] = value
                combinations.append(tuple(row))
    else:
        options = [
            list(dict.fromkeys(row[index] for row in task["values"]))
            for index in range(len(fields))
        ]
        combinations = list(itertools.product(*options))
    if len(combinations) < 100:
        raise ValueError(f"not enough distinct values for {task['tool']}")
    indices = [round(i * (len(combinations) - 1) / 99) for i in range(100)]
    return [combinations[index] for index in indices]


def generate_rows() -> list[dict]:
    rows = []
    tools = {task["tool"]: _tool(task) for task in TASKS}
    task_names = [task["tool"] for task in TASKS]
    for task_index, task in enumerate(TASKS):
        tool = _tool(task)
        argument_combinations = _value_combinations(task)
        split_tools = [
            name
            for name in task_names
            if (name in VALIDATION_TOOLS) == (task["tool"] in VALIDATION_TOOLS)
        ]
        competitor_names = [name for name in split_tools if name != task["tool"]]
        for dialect in DIALECTS:
            dialect_index = DIALECTS.index(dialect)
            for value_index, values in enumerate(argument_combinations):
                form_index = value_index % len(PROMPT_FORMS[dialect])
                form = PROMPT_FORMS[dialect][form_index]
                competitor_index = (dialect_index * 100 + value_index) % len(
                    competitor_names
                )
                competitor = tools[competitor_names[competitor_index]]
                arguments = dict(zip(task["fields"], values, strict=True))
                details = "، ".join(
                    f"{label}: {_prompt_value(field, value, form_index)}"
                    for field, label, value in zip(
                        task["fields"][1:],
                        task["labels"][1:],
                        values[1:],
                        strict=True,
                    )
                )
                utterance = form.format(
                    subject=task["request_subject"].format(value=values[0]),
                    place=task["place"],
                    details=details,
                )
                if form_index == 17:
                    utterance = _prompt_value("", utterance, form_index)
                index = len(rows) + 1
                tool_call = {
                    "type": "function",
                    "function": {
                        "name": task["tool"],
                        "arguments": arguments,
                    },
                }
                rows.append(
                    {
                        "id": f"SILA-TR-{index:04d}",
                        "language": "ar",
                        "dialect": dialect,
                        "domain": task["tool"],
                        "split": "validation"
                        if task["tool"] in VALIDATION_TOOLS
                        else "train",
                        "user_utterance": utterance,
                        "available_tools": [tool, competitor]
                        if (dialect_index + value_index + task_index) % 2 == 0
                        else [competitor, tool],
                        "messages": [
                            {"role": "user", "content": utterance},
                            {"role": "assistant", "tool_calls": [tool_call]},
                        ],
                        "should_call_tool": True,
                        "expected_tool_name": task["tool"],
                        "expected_arguments": arguments,
                        "generation_coordinates": {
                            "task": task["tool"],
                            "dialect": dialect,
                            "form": form_index,
                            "values": value_index,
                        },
                    }
                )
    for dialect_index, dialect in enumerate(DIALECTS):
        for task_index, (domain, response_template) in enumerate(NO_CALL_TASKS):
            for form_index in range(10):
                split = "validation" if form_index >= 8 else "train"
                values = NO_CALL_VALUE_SETS[task_index][form_index]
                request_template = (
                    NO_CALL_MISSING_REQUESTS[split][dialect]
                    if domain == "missing_required_input"
                    else NO_CALL_REQUESTS[dialect][task_index]
                )
                split_tools = [
                    name
                    for name in task_names
                    if (name in VALIDATION_TOOLS) == (split == "validation")
                ]
                request = request_template.format(**values)
                utterance = NO_CALL_FORMS[dialect][form_index].format(request=request)
                if request.endswith("؟"):
                    utterance = utterance.removesuffix(".")
                answer = (
                    NO_CALL_MISSING_ANSWERS[split][dialect]
                    if domain == "missing_required_input"
                    else NO_CALL_OUT_OF_SCOPE_ANSWERS[dialect].format(**values)
                    if domain == "out_of_scope"
                    else response_template.format(**values)
                )
                pair_start = (dialect_index * 19 + task_index * 7 + form_index) % len(
                    split_tools
                )
                if domain == "missing_required_input":
                    relevant = (
                        "find_calligraphy_workshop"
                        if split == "train"
                        else "find_public_lecture"
                    )
                    distractor_names = [n for n in split_tools if n != relevant]
                    available_tools = [
                        tools[relevant],
                        tools[distractor_names[pair_start % len(distractor_names)]],
                    ]
                else:
                    available_tools = [
                        tools[split_tools[pair_start]],
                        tools[split_tools[(pair_start + 1) % len(split_tools)]],
                    ]
                if (dialect_index + task_index + form_index) % 2:
                    available_tools.reverse()
                index = len(rows) + 1
                rows.append(
                    {
                        "id": f"SILA-TR-{index:04d}",
                        "language": "ar",
                        "dialect": dialect,
                        "domain": domain,
                        "split": split,
                        "user_utterance": utterance,
                        "available_tools": available_tools,
                        "messages": [
                            {"role": "user", "content": utterance},
                            {"role": "assistant", "content": answer},
                        ],
                        "should_call_tool": False,
                        "expected_tool_name": None,
                        "expected_arguments": None,
                        "generation_coordinates": {
                            "task": domain,
                            "dialect": dialect,
                            "form": form_index,
                            "values": form_index,
                        },
                    }
                )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    path.write_bytes(content)
    return content


def generate(output_dir: Path) -> dict:
    rows = generate_rows()
    positive_rows = [row for row in rows if row["expected_tool_name"] is not None]
    no_call_rows = [row for row in rows if row["expected_tool_name"] is None]
    if len(positive_rows) != 6000 or len(no_call_rows) != 600:
        raise ValueError("expected 6,000 calls and 600 no-call examples")
    if len(rows) != 6600 or len({row["user_utterance"] for row in rows}) != 6600:
        raise ValueError("expected 6,600 unique synthetic prompts")
    if {row["language"] for row in rows} != {"ar"}:
        raise ValueError("training corpus must contain only Arabic rows")
    if {row["dialect"] for row in rows} != set(DIALECTS):
        raise ValueError("training corpus must include all six Arabic dialects")
    if any(len(_value_combinations(task)) != 100 for task in TASKS):
        raise ValueError("each tool and dialect must have 100 argument combinations")
    if len(NO_CALL_VALUE_SETS) != len(NO_CALL_TASKS) or any(
        len(values) != 10
        or len({json.dumps(value, sort_keys=True) for value in values}) != 10
        for values in NO_CALL_VALUE_SETS
    ):
        raise ValueError("each no-call intent must have ten distinct content variants")
    if any(
        len(forms) != 20 or len(set(forms)) != 20 for forms in PROMPT_FORMS.values()
    ):
        raise ValueError("each dialect must have 20 distinct positive prompt forms")
    if any(
        _prompt_value(field, value, row["generation_coordinates"]["form"])
        not in row["user_utterance"]
        for row in positive_rows
        for field, value in row["expected_arguments"].items()
    ):
        raise ValueError("every expected argument must appear in its user prompt")
    if any(
        not row["should_call_tool"]
        or row["expected_tool_name"]
        not in {tool["name"] for tool in row["available_tools"]}
        for row in positive_rows
    ):
        raise ValueError("expected tool must be available in its prompt")
    if any(
        len(row["available_tools"]) != 2
        or row["available_tools"][0]["name"] == row["available_tools"][1]["name"]
        for row in positive_rows
    ):
        raise ValueError("every prompt must offer two distinct tools")
    for row in positive_rows:
        if (
            row["messages"][0] != {"role": "user", "content": row["user_utterance"]}
            or row["messages"][1]["role"] != "assistant"
            or row["messages"][1]["tool_calls"][0]["function"]["name"]
            != row["expected_tool_name"]
            or row["messages"][1]["tool_calls"][0]["function"]["arguments"]
            != row["expected_arguments"]
        ):
            raise ValueError("Qwen message target must match the expected call")
    if any(
        row["messages"]
        != [
            {"role": "user", "content": row["user_utterance"]},
            {"role": "assistant", "content": row["messages"][1].get("content")},
        ]
        or not row["messages"][1].get("content")
        or row["should_call_tool"]
        or row["expected_tool_name"] is not None
        or row["expected_arguments"] is not None
        for row in no_call_rows
    ):
        raise ValueError("no-call rows must have a plain assistant response")
    pair_counts = Counter(
        (
            row["expected_tool_name"],
            next(
                tool["name"]
                for tool in row["available_tools"]
                if tool["name"] != row["expected_tool_name"]
            ),
        )
        for row in positive_rows
    )
    for task in TASKS:
        name = task["tool"]
        expected_choices = 1 if name in VALIDATION_TOOLS else 7
        competitors = [other for target, other in pair_counts if target == name]
        counts = [pair_counts[(name, other)] for other in competitors]
        if len(competitors) != expected_choices or max(counts) - min(counts) > 1:
            raise ValueError(f"unbalanced distractor coverage for {name}")
    positions = Counter(
        (row["expected_tool_name"], row["available_tools"][0]["name"]) for row in rows
    )
    for task in TASKS:
        name = task["tool"]
        if positions[(name, name)] != 300:
            raise ValueError(f"unbalanced tool position for {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    task_by_tool = {task["tool"]: task for task in TASKS}
    prompt_feature_counts = {
        "orthographic_hamza_variation": sum(
            row["generation_coordinates"]["form"] == 17 for row in positive_rows
        ),
        "code_switching": sum(
            row["generation_coordinates"]["form"] == 18 for row in positive_rows
        ),
        "normalized_date_surface": sum(
            row["generation_coordinates"]["form"] == 19
            and any(
                field.endswith("_date_iso")
                for field in task_by_tool[row["expected_tool_name"]]["fields"]
            )
            for row in positive_rows
        ),
    }
    if prompt_feature_counts != {
        "orthographic_hamza_variation": 300,
        "code_switching": 300,
        "normalized_date_surface": 60,
    }:
        raise ValueError("prompt variation counts do not match the design")
    train_bytes = _write_jsonl(output_dir / "training.train.jsonl", train)
    validation_bytes = _write_jsonl(
        output_dir / "training.validation.jsonl", validation
    )
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {
        "dataset": "SILA first-party synthetic Arabic tool-calling corpus",
        "training_format": "Qwen messages with function calls and per-row tools",
        "format_model": QWEN_MODEL,
        "format_model_revision": QWEN_REVISION,
        "generator": "sila.generate_training_corpus",
        "generator_source_file": "src/sila/generate_training_corpus.py",
        "generator_version": GENERATOR_VERSION,
        "generator_sha256": generator_hash,
        "python_version": platform.python_version(),
        "external_sources": [],
        "external_generators": [],
        "license": "unassigned; generated in this repository without upstream data",
        "language": "ar",
        "dialect_counts": {
            dialect: sum(row["dialect"] == dialect for row in rows)
            for dialect in DIALECTS
        },
        "positive_rows": len(positive_rows),
        "no_call_rows": len(no_call_rows),
        "no_call_dialect_counts": {
            dialect: sum(row["dialect"] == dialect for row in no_call_rows)
            for dialect in DIALECTS
        },
        "linguistic_review": "not performed",
        "row_count": len(rows),
        "tool_count": len(TASKS),
        "argument_combinations_per_tool_dialect": 100,
        "prompt_feature_counts": prompt_feature_counts,
        "dialect_split_counts": {
            split: {
                dialect: sum(
                    row["dialect"] == dialect and row["split"] == split for row in rows
                )
                for dialect in DIALECTS
            }
            for split in ("train", "validation")
        },
        "split_policy": {
            "method": (
                "whole-tool call holdout; no-call form and content holdout; "
                "deterministic, no random seed"
            ),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "validation_tools": sorted(VALIDATION_TOOLS),
        },
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "validation_sha256": hashlib.sha256(validation_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(train_bytes + validation_bytes).hexdigest(),
    }
    manifest_path = output_dir / "training.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
