import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
import io, contextlib, traceback
from google import genai
from google.genai import types
from streamlit_float import float_init, float_css_helper

def require_username_only():
    """บังคับให้กรอก Username ก่อนเข้าใช้งาน (ไม่ใช่ระบบยืนยันตัวตน)"""
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""

    # compatibility (ถ้าโค้ดเดิมเคยใช้ key username)
    if "username" not in st.session_state:
        st.session_state.username = st.session_state.user_name

    # ผ่านแล้ว
    if (st.session_state.user_name or "").strip():
        st.session_state.username = st.session_state.user_name.strip()
        return

    st.title("👤 กรุณากรอก Username ก่อนเข้าใช้งาน")
    st.caption("ใช้เพื่อผูกบริบท/บันทึกคะแนนในแอปนี้เท่านั้น (ไม่ใช่ระบบล็อกอินจริง)")

    with st.form("username_gate"):
        u = st.text_input("Username", placeholder="เช่น Sunanta / Student01")
        ok = st.form_submit_button("เริ่มใช้งาน")

    if ok:
        u = (u or "").strip()
        if u:
            st.session_state.user_name = u
            st.session_state.username = u
            st.rerun()
        else:
            st.error("กรุณากรอก Username")

    st.stop()

@st.cache_resource
def get_gemini_client():
    return genai.Client()  # จะอ่าน GEMINI_API_KEY / GOOGLE_API_KEY จาก env ได้ :contentReference[oaicite:4]{index=4}

def gemini_reply(messages: list[dict], user_text: str, ctx: dict) -> str:
    # ทำ transcript สั้น ๆ
    transcript = []
    for m in messages[-20:]:
        transcript.append(f"{m['role'].upper()}: {m['content']}")
    transcript_text = "\n".join(transcript)

    system_inst = (
        "คุณคือผู้ช่วยสอน Python ในแอป นี้ "
        "ต้องตอบเป็นภาษาไทยเป็นหลัก, กระชับ, อ้างอิงบริบท (page/lesson) ที่ให้มา "
        "ถ้าอยู่หน้า Lessons ให้สอนตามบทนั้นและยกตัวอย่างโค้ดสั้น ๆ "
        "ถ้าอยู่หน้า Quiz ให้ช่วยอธิบายแนวคิด/วิธีคิด ไม่เฉลยถ้าผู้ใช้ไม่ขอ"
    )

    prompt = (
        f"APP_CONTEXT_JSON:\n{json.dumps(ctx, ensure_ascii=False)}\n\n"
        f"CHAT_TRANSCRIPT:\n{transcript_text}\n\n"
        f"USER:\n{user_text}"
    )

    client = get_gemini_client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_inst
        ),
    )
    return resp.text or "(ไม่มีข้อความตอบกลับ)"
def _build_prompt(ctx: dict, messages: list[dict], user_text: str) -> str:
    # เก็บประวัติแชทสั้น ๆ กัน prompt ยาวเกิน
    transcript = []
    for m in messages[-20:]:
        transcript.append(f"{m['role'].upper()}: {m['content']}")
    transcript_text = "\n".join(transcript)

    return (
        "APP_CONTEXT_JSON:\n"
        f"{json.dumps(ctx or {}, ensure_ascii=False)}\n\n"
        "CHAT_TRANSCRIPT:\n"
        f"{transcript_text}\n\n"
        "USER:\n"
        f"{user_text}"
    )

def _ask_gemini(ctx: dict, messages: list[dict], user_text: str) -> str:
    system_inst = (
        "คุณคือผู้ช่วยสอน Python ในแอป Streamlit นี้\n"
        "- ตอบภาษาไทยเป็นหลัก กระชับ เข้าใจง่าย\n"
        "- อ้างอิงบริบทจาก APP_CONTEXT_JSON เสมอ (เช่น page/lesson)\n"
        "- ถ้า page=Lessons ให้สอนตาม lesson_title/lesson_excerpt และยกตัวอย่างโค้ดสั้น ๆ\n"
        "- ถ้า page=Quiz ให้ช่วยอธิบายแนวคิด/วิธีคิด และจะไม่เฉลยตรง ๆ จนกว่าผู้ใช้จะขอ\n"
        "- ถ้า page=Dashboard ให้ช่วยอ่านสถิติและแนะนำสิ่งที่ควรทบทวน"
    )

    client = get_gemini_client()
    
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=_build_prompt(ctx, messages, user_text),
        config=types.GenerateContentConfig(system_instruction=system_inst),
    )
    return resp.text or "(ไม่มีข้อความตอบกลับ)"

def corner_chat():
    # state
    if "corner_chat_open" not in st.session_state:
        st.session_state.corner_chat_open = False
    if "corner_chat_msgs" not in st.session_state:
        st.session_state.corner_chat_msgs = []
    if "corner_chat_text" not in st.session_state:
        st.session_state.corner_chat_text = ""

    # ===== 1) ปุ่มลอยมุมขวาล่าง =====
    fab = st.container()
    with fab:
        if st.button("💬", key="corner_chat_fab", help="เปิด/ปิดแชท"):
            st.session_state.corner_chat_open = not st.session_state.corner_chat_open
            st.rerun()

    fab.float(float_css_helper(right="1rem", bottom="1rem", width="3.2rem"))

    # ===== 2) กล่องแชทลอย =====
    if st.session_state.corner_chat_open:
        box = st.container()
        with box:
            top = st.columns([1, 1])
            with top[0]:
                st.markdown("**💬 Chat**")
            with top[1]:
                if st.button("✖ ปิด", key="corner_chat_close"):
                    st.session_state.corner_chat_open = False
                    st.rerun()

            # แสดงข้อความย้อนหลัง
            for m in st.session_state.corner_chat_msgs:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

                    # ---------- callbacks ----------
            if "corner_chat_to_send" not in st.session_state:
                st.session_state.corner_chat_to_send = None

            def _queue_send():
                text = (st.session_state.corner_chat_text or "").strip()
                if text:
                    st.session_state.corner_chat_to_send = text
                    st.session_state.corner_chat_text = ""  # เคลียร์ input แบบปลอดภัย (callback)

            def _clear_chat():
                st.session_state.corner_chat_msgs = []
                st.session_state.corner_chat_text = ""
                st.session_state.corner_chat_to_send = None

            # input + buttons
            st.text_input("พิมพ์ข้อความ…", key="corner_chat_text")
            c1, c2 = st.columns([1, 1])
            with c1:
                st.button("ส่ง", key="corner_chat_send", use_container_width=True, on_click=_queue_send)
            with c2:
                st.button("ล้างแชท", key="corner_chat_clear", use_container_width=True, on_click=_clear_chat)

            # ถ้ามีข้อความที่ถูกคิวไว้ -> ค่อยเรียก LLM
            if st.session_state.corner_chat_to_send:
                user_text = st.session_state.corner_chat_to_send
                st.session_state.corner_chat_to_send = None

                st.session_state.corner_chat_msgs.append({"role": "user", "content": user_text})

                ctx = st.session_state.get("app_ctx", {"page": "unknown"})
                with st.spinner("กำลังคิด..."):
                    ans = _ask_gemini(ctx, st.session_state.corner_chat_msgs, user_text)

                st.session_state.corner_chat_msgs.append({"role": "assistant", "content": ans})
                st.rerun()
            box.float(
                    float_css_helper(
                        right="1rem",
                        bottom="5.2rem",
                        width="380px",
                        padding="0.75rem",
                        border="1px solid rgba(255,255,255,0.22)",
                        background="#000000",
                    )
                    + "max-height: 65vh; overflow: auto;"
            )

# ============================
# จัดการไฟล์สถิติ
# ============================
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "history.json")

def load_history():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_history(history):
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"บันทึกข้อมูลไม่ได้: {e}")

# ============================
# บทเรียนทั้งหมด (ตั้งแต่พื้นฐานถึง OOP) — ฉบับละเอียด
# ============================
lessons = {
    "print": {
        "title": "การแสดงผลด้วย print()",
        "content": """## 🎯 เป้าหมาย
- ใช้ `print()` ได้ทั้งพื้นฐานและขั้นสูง (sep/end, f-string)
- จัดรูปแบบข้อความ/ตัวเลข/ตัวแปรอย่างถูกต้อง

## 🧠 แนวคิดหลัก
- รูปแบบ: `print(*objects, sep=' ', end='\\n')`
- ใช้ **f-string**: `f"sum={x+y}"` เร็ว/อ่านง่ายกว่าต่อสตริง

## 💡 ตัวอย่าง
```python
x, y = 7, 3
print("Hello", "Python", sep=" | ")  # Hello | Python
print("x =", x, end="; ")            # x = 7; 
print(f"y = {y}, sum = {x+y}")       # y = 3, sum = 10

# จัดรูปแบบทศนิยม
pi = 3.1415926
print(f"{pi:.2f}")  # 3.14
```

## ⚠️ Pitfalls
- ลืม `f` หน้า string เมื่อใช้ `{}` → ไม่แทนค่า
- ใส่ `end` ไม่เหมาะสมทำให้บรรทัดต่อไปติดกัน

## 🧪 Try it
- พิมพ์สูตรคูณแม่ 5 ในบรรทัดเดียว คั่นด้วยเครื่องหมายจุลภาค
""",
        "quiz": [
            {"question": "อาร์กิวเมนต์ใดของ print ใช้เปลี่ยนตัวคั่นค่า?", "choices": ["end", "sep", "fmt"], "answer": "sep"},
            {"question": "f-string ต้องขึ้นต้นด้วยอะไร?", "choices": ["s", "f", "r"], "answer": "f"},
        ],
    },

    "variables": {
        "title": "ตัวแปร (Variables)",
        "content": """## 🎯 เป้าหมาย
- ตั้งชื่อตัวแปรให้สื่อความหมายและถูกกฎ (PEP8)
- เข้าใจ dynamic typing และ name binding

## 🧠 แนวคิดหลัก
- ตั้งชื่อ: a-z, A-Z, 0-9, `_` (ห้ามขึ้นต้นด้วยตัวเลข)
- Python เป็น dynamically typed (ชนิดตามค่าล่าสุด)
- ชื่อควร snake_case เช่น `total_score`, `first_name`

## 💡 ตัวอย่าง
```python
count = 10      # int
count = "ten"   # ตอนนี้กลายเป็น str
first_name = "Alice"
_total = 99
```

## ⚠️ Pitfalls
- ใช้ชื่อชนกับคำสงวน (`class`, `def`, `for`, …)
- เปลี่ยนชนิดโดยไม่ตั้งใจ → บั๊กตามยาก

## 🧪 Try it
- สร้างตัวแปร 3 ตัว: ชื่อ, อายุ, จังหวัด แล้วพิมพ์ด้วย f-string 1 บรรทัด
""",
        "quiz": [
            {"question": "ชื่อตัวแปรใดถูกต้องตามกฎ?", "choices": ["2name", "first_name", "class"], "answer": "first_name"},
        ],
    },

    "datatypes": {
        "title": "ชนิดข้อมูลพื้นฐาน (Data Types)",
        "content": """## 🎯 เป้าหมาย
- แยกความต่างของ `int`, `float`, `str`, `bool` ได้ชัดเจน
- เข้าใจ truthy/falsey

## 🧠 แนวคิดหลัก
- `bool(0)==False`, `bool(1)==True`, `bool(\"\")==False`, `bool(\"x\")==True`
- `type(x)` ตรวจชนิด / `isinstance(x, T)` เช็กชนิดหลายทาง

## 💡 ตัวอย่าง
```python
values = [0, 1, "", "a", [], [1]]
for v in values:
    print(repr(v), "=>", bool(v))
```

## ⚠️ Pitfalls
- เทียบ `==` กับ `is` คนละอย่าง: `==` เทียบค่า, `is` เทียบอ้างอิงวัตถุ
- ระวัง float precision เช่น `0.1 + 0.2 != 0.3` เป๊ะ ๆ

## 🧪 Try it
- ออกแบบลิสต์ 6 ค่าแล้วพิมพ์ `bool(v)` พร้อมอธิบายผลแต่ละตัว
""",
        "quiz": [
            {"question": "ค่าต่อไปนี้ falsey คืออะไร?", "choices": ["0", "[]", '\"\"', "ทั้งหมด"], "answer": "ทั้งหมด"},
        ],
    },

    "casting": {
        "title": "การแปลงชนิดข้อมูล (Casting)",
        "content": """## 🎯 เป้าหมาย
- แปลงชนิดข้อมูลได้ถูกต้อง ป้องกัน error

## 🧠 แนวคิดหลัก
- `int(x)`, `float(x)`, `str(x)`, `bool(x)`
- ข้อความที่แปลงเป็นตัวเลขต้องเป็นรูปแบบตัวเลขเท่านั้น

## 💡 ตัวอย่าง
```python
a = "123"; b = int(a)
c = float("3.14"); d = str(100)
# bool: ว่าง/ศูนย์เป็น False
print(bool(""), bool(0), bool("ok"))
```

## ⚠️ Pitfalls
- `int("12a")` จะ error
- แปลง float → int จะปัดทิ้งส่วนทศนิยม

## 🧪 Try it
- รับอินพุตเป็นสตริงตัวเลข แล้วพิมพ์ int, float และชนิดของมัน
""",
        "quiz": [
            {"question": "ข้อใดแปลงได้โดยไม่ error?", "choices": ['int("456")', 'int("45.6")', 'float("45a")'], "answer": 'int("456")'},
        ],
    },

    "strings": {
        "title": "Strings (ข้อความ)",
        "content": """## 🎯 เป้าหมาย
- ใช้ indexing, slicing, methods สำคัญ, f-string
- เข้าใจ immutability ของสตริง

## 🧠 แนวคิดหลัก
- สตริงแก้ตรง ๆ ไม่ได้ (immutable) → สร้างใหม่
- เมธอดสำคัญ: `upper`, `lower`, `strip`, `replace`, `split`, `join`, `startswith`, `endswith`, `find`

## 💡 ตัวอย่าง
```python
s = "  hello Python  "
print(s.strip().title())            # Hello Python
print(" - ".join(["A","B","C"]))    # A - B - C
name, score = "Alice", 92.567
print(f"{name}: {score:.1f}")       # Alice: 92.6
```

## ⚠️ Pitfalls
- ต่อสตริงในลูปด้วย `+` มาก ๆ → ช้า ควรสะสมในลิสต์แล้ว `join`
- ลืม `.strip()` ตอนอ่านอินพุต → ช่องว่างแฝง

## 🧪 Try it
- รับชื่อเต็ม “ชื่อ นามสกุล” แล้วพิมพ์ “นามสกุล, ชื่อ” ด้วย `.split()` + f-string
""",
        "quiz": [
            {"question": "สตริงเป็นชนิดใด?", "choices": ["mutable", "immutable"], "answer": "immutable"},
        ],
    },

    "booleans_operators": {
        "title": "Boolean และ Operators",
        "content": """## 🎯 เป้าหมาย
- ใช้เปรียบเทียบ/ตรรกะได้คล่อง
- เข้าใจ short-circuit ของ `and`, `or`

## 🧠 แนวคิดหลัก
- `and` คืนค่าซ้ายถ้า falsey ไม่งั้นคืนขวา
- `or` คืนค่าซ้ายถ้า truthy ไม่งั้นคืนขวา
- เปรียบเทียบลูกโซ่: `0 < x < 10`

## 💡 ตัวอย่าง
```python
x = 0 or "fallback"   # "fallback"
y = "" or "N/A"       # "N/A"
z = "ok" and 123      # 123
print(3 < 5 < 10)     # True
```

## ⚠️ Pitfalls
- ใช้ `=` แทน `==` ใน if (Python จะ error)
- เทียบสตริงต่างตัวพิมพ์ → ใช้ `.lower()` ช่วย

## 🧪 Try it
- เขียนตัวกรอง: ถ้าอินพุตว่างให้แทนเป็น "N/A" ด้วย `or`
""",
        "quiz": [
            {"question": "`'' or 'x'` คืนค่าอะไร?", "choices": ["''", "'x'", "False"], "answer": "'x'"},
        ],
    },

    "lists": {
        "title": "List (ลิสต์)",
        "content": """## 🎯 เป้าหมาย
- เข้าใจลิสต์ (mutable), slicing, list comprehension
- ใช้เมธอด: `append`, `extend`, `insert`, `remove`, `pop`, `sort`, `reverse`

## 🧠 แนวคิดหลัก
- copy/shallow vs deep copy
- `list.sort()` (in-place) vs `sorted(list)` (คืนใหม่)

## 💡 ตัวอย่าง
```python
nums = [1, 2, 3]
nums2 = nums            # อ้างอิงเดียวกัน
nums_copy = nums[:]     # copy ใหม่
nums.append(4)
print(nums, nums2)      # ทั้งสองเปลี่ยน
print(nums_copy)        # สำเนาเดิม

squares = [n*n for n in range(1,6) if n%2==1]  # [1, 9, 25]
```

## ⚠️ Pitfalls
- สับสนระหว่างอ้างอิงเดียวกับการคัดลอก
- เปลี่ยนขณะวนลูป → เก็บคีย์ไว้ก่อน

## 🧪 Try it
- จากลิสต์ตัวเลข สร้างลิสต์ “เลขคู่ยกกำลังสอง” ด้วย comprehension
""",
        "quiz": [
            {"question": "วิธีคัดลอกลิสต์อย่างเร็ว?", "choices": ["l2 = l1", "l2 = l1[:]", "l2 = copy"], "answer": "l2 = l1[:]"},
        ],
    },

    "tuples": {
        "title": "Tuple (ทูเพิล)",
        "content": """## 🎯 เป้าหมาย
- เข้าใจ tuple ว่า immutable ใช้เก็บค่าคงที่

## 🧠 แนวคิดหลัก
- ใช้ `()` และรองรับ unpack
- เร็วและปลอดภัยกว่า list เมื่อไม่ต้องแก้ไข

## 💡 ตัวอย่าง
```python
t = (1, 2, 3)
a, b, c = t
print(a, b, c)
```

## ⚠️ Pitfalls
- ต้องมี comma เมื่อเป็น single item: `t = (1,)`

## 🧪 Try it
- ทำการสลับค่าตัวแปรสองตัวด้วย tuple unpacking
""",
        "quiz": [
            {"question": "tuple แก้ไขค่าได้หรือไม่?", "choices": ["ได้", "ไม่ได้"], "answer": "ไม่ได้"},
        ],
    },

    "sets": {
        "title": "Set (เซต)",
        "content": """## 🎯 เป้าหมาย
- ใช้ set สำหรับค่าที่ไม่ซ้ำ และปฏิบัติการเชิงเซต

## 🧠 แนวคิดหลัก
- ไม่มีลำดับ, ไม่เก็บค่าซ้ำ
- ปฏิบัติการ: union `|`, intersect `&`, diff `-`, symdiff `^`

## 💡 ตัวอย่าง
```python
a, b = {1,2,3}, {3,4,5}
print(a | b, a & b, a - b, a ^ b)  # {1,2,3,4,5} {3} {1,2} {1,2,4,5}
```

## ⚠️ Pitfalls
- สมาชิกต้อง hashable (ห้าม list/dict)

## 🧪 Try it
- จากลิสต์ที่มีค่าซ้ำ ให้แปลงเป็น set เพื่อคัดค่าซ้ำทิ้ง
""",
        "quiz": [
            {"question": "ผลของ {1,2,2,3} คือ?", "choices": ["{1,2,2,3}", "{1,2,3}", "{2,3}"], "answer": "{1,2,3}"},
        ],
    },

    "dictionaries": {
        "title": "Dictionary (ดิกชันนารี)",
        "content": """## 🎯 เป้าหมาย
- ใช้ dict สำหรับ key→value อย่างถูกต้อง
- เมธอด: `get`, `keys`, `values`, `items`, `update`, `pop`

## 🧠 แนวคิดหลัก
- key ต้อง hashable (str,int,tuple-immutable)
- `get(k, default)` ป้องกัน KeyError

## 💡 ตัวอย่าง
```python
person = {"name": "Alice", "age": 25}
print(person.get("city", "Unknown"))
for k, v in person.items():
    print(k, "=>", v)
```

## ⚠️ Pitfalls
- ลบคีย์ระหว่างวนลูปใน dict เดิม → เก็บก่อนค่อยลบ

## 🧪 Try it
- นับความถี่อักษรในสตริงหนึ่งบรรทัดและพิมพ์ตารางผล
""",
        "quiz": [
            {"question": "ชนิดใดห้ามเป็น key?", "choices": ["str", "int", "list"], "answer": "list"},
        ],
    },

    "if_else": {
        "title": "เงื่อนไข if / elif / else",
        "content": """## 🎯 เป้าหมาย
- ออกแบบเงื่อนไขซ้อน/หลายทางได้ดี
- ใช้ ternary expression

## 🧠 แนวคิดหลัก
- เรียงจากเฉพาะ → ทั่วไป
- ternary: `a if cond else b`

## 💡 ตัวอย่าง
```python
score = 82
grade = ("A" if score>=80 else "B") if score>=70 else "C"
print(grade)  # A

x = -5
if x > 0:
    print("positive")
elif x == 0:
    print("zero")
else:
    print("negative")
```

## ⚠️ Pitfalls
- เงื่อนไขทับซ้อน/ซ้ำซ้อน
- ใช้ `== True` โดยไม่จำเป็น

## 🧪 Try it
- เขียนตัวจัดเกรด A/B/C/D/F ช่วงคะแนน 0–100 อย่างรัดกุม
""",
        "quiz": [
            {"question": "นิพจน์ใดคือ ternary?", "choices": ["a if cond else b", "if a: b", "cond ? a : b"], "answer": "a if cond else b"},
        ],
    },

    "while_loop": {
        "title": "ลูป while",
        "content": """## 🎯 เป้าหมาย
- ใช้ while อย่างปลอดภัย ไม่เกิดลูปไม่รู้จบ
- เข้าใจ break/continue

## 🧠 แนวคิดหลัก
- while ทำงาน “ตราบใดที่เงื่อนไขเป็น True”
- ต้องเปลี่ยนเงื่อนไขในลูปเสมอ

## 💡 ตัวอย่าง
```python
count = 0
while count < 3:
    print("รอบ", count)
    count += 1
```

## ⚠️ Pitfalls
- ลืมเปลี่ยนค่า → ลูปไม่รู้จบ

## 🧪 Try it
- เขียนลูป while รับอินพุตจนกว่าจะพิมพ์คำว่า "exit"
""",
        "quiz": [
            {"question": "while หยุดทำงานเมื่อใด?", "choices": ["เงื่อนไขเป็น True", "เงื่อนไขเป็น False"], "answer": "เงื่อนไขเป็น False"},
        ],
    },

    "for_loop": {
        "title": "ลูป for",
        "content": """## 🎯 เป้าหมาย
- ใช้ for กับ range/list/string/dict, enumerate, zip
- เข้าใจ break/continue

## 🧠 แนวคิดหลัก
- `range(start, stop, step)` (ไม่รวม stop)
- `enumerate(seq, start=1)` ได้ index+ค่า
- `zip(a, b)` จับคู่สมาชิก

## 💡 ตัวอย่าง
```python
for i in range(2, 10, 2):  # 2,4,6,8
    print(i, end=" ")

fruits = ["apple", "banana", "cherry"]
for idx, name in enumerate(fruits, start=1):
    print(idx, name)

a, b = [1,2,3], ["one","two","three"]
for n, word in zip(a, b):
    print(n, "=>", word)
```

## ⚠️ Pitfalls
- เปลี่ยนความยาวโครงสร้างขณะวนลูป

## 🧪 Try it
- ใช้ zip รวมชื่อวิชาและคะแนน แล้วพิมพ์ “วิชา:คะแนน”
""",
        "quiz": [
            {"question": "range(4) ให้ค่าใดบ้าง?", "choices": ["0–3", "1–4", "0–4"], "answer": "0–3"},
        ],
    },

    "functions": {
        "title": "ฟังก์ชัน (Functions)",
        "content": """## 🎯 เป้าหมาย
- เขียนฟังก์ชันรับพารามิเตอร์/คืนค่าถูกต้อง
- เข้าใจ default args, *args, **kwargs, scope

## 🧠 แนวคิดหลัก
- default args: ระวังชนิด mutable
- `*args` รับตามลำดับไม่จำกัด, `**kwargs` ตามชื่อ
- ช่วงชีวิตตัวแปร (local/global)

## 💡 ตัวอย่าง
```python
def add(a, b=0): return a + b
def total(*nums): return sum(nums)
def show(**info): return info

print(add(5))         # 5
print(total(1,2,3))   # 6
print(show(name="Alice", age=25))  # {'name':'Alice','age':25}
```

## ⚠️ Pitfalls
- default mutable → สะสมค้าง
- shadowing: ตัวแปรชื่อซ้ำทำให้งง

## 🧪 Try it
- เขียนฟังก์ชัน `flatten(list_of_lists)` คืน list เดียวจากลิสต์ซ้อน
""",
        "quiz": [
            {"question": "คำสำคัญประกาศฟังก์ชันคือ?", "choices": ["function", "def", "fun"], "answer": "def"},
            {"question": "คืนค่าจากฟังก์ชันด้วยคำว่า?", "choices": ["back", "return", "output"], "answer": "return"},
        ],
    },

    "classes": {
        "title": "คลาสและวัตถุ (Class & Object / OOP เบื้องต้น)",
        "content": """## 🎯 เป้าหมาย
- เข้าใจ class/instance attribute, method, constructor
- ออกแบบคลาสง่าย ๆ ใช้งานได้จริง

## 🧠 แนวคิดหลัก
- `class` = พิมพ์เขียว, `object` = อินสแตนซ์จริง
- `__init__` เรียกตอนสร้าง object (constructor)
- method: instance (`self`), class (`@classmethod`), static (`@staticmethod`)

## 💡 ตัวอย่าง
```python
class Counter:
    total = 0  # class attribute ใช้ร่วมกันทุก object

    def __init__(self, start=0):
        self.value = start  # instance attribute เฉพาะแต่ละ object

    def inc(self, step=1):
        self.value += step
        Counter.total += step

    @classmethod
    def get_total(cls):
        return cls.total

    @staticmethod
    def is_even(n):
        return n % 2 == 0

c1 = Counter()
c2 = Counter(10)
c1.inc(); c2.inc(5)
print(c1.value, c2.value)    # 1, 15
print(Counter.get_total())   # 6
print(Counter.is_even(10))   # True
```

## ⚠️ Pitfalls
- เขียน `Counter.value` แทน `self.value` (ไปแก้ที่คลาส ไม่ใช่อินสแตนซ์)
- สับสน `@classmethod` vs `@staticmethod`

## 🧪 Try it
- สร้างคลาส `BankAccount(owner)` มี `deposit`, `withdraw`, `balance` และป้องกันถอนเกิน
""",
        "quiz": [
            {"question": "อะไรต่างกัน: class vs instance attribute?",
             "choices": ["class ใช้ร่วมกันทุก object / instance เฉพาะแต่ละ object",
                         "instance ใช้ร่วมกันทุก object / class เฉพาะแต่ละ object",
                         "ทั้งคู่เหมือนกัน"],
             "answer": "class ใช้ร่วมกันทุก object / instance เฉพาะแต่ละ object"},
            {"question": "เมธอดใดไม่ต้องรับ self?", "choices": ["instance method", "classmethod", "staticmethod"], "answer": "staticmethod"},
            {"question": "constructor คือเมธอดใด?", "choices": ["__call__", "__repr__", "__init__"], "answer": "__init__"},
        ],
    },
}
def set_app_context(page: str, user: str, lesson_key: str | None = None, extra: dict | None = None):
    ctx = {
        "page": page,
        "user": user or "(ไม่ระบุ)",
        "lesson_key": lesson_key,
        "lesson_title": lessons[lesson_key]["title"] if lesson_key in lessons else None,
    }
    if extra:
        ctx.update(extra)
    st.session_state.app_ctx = ctx
# ============================
# แอปหลัก Streamlit
# ============================
st.set_page_config(page_title="Python Learning App — Detailed", layout="wide")
float_init()
require_username_only()
st.sidebar.title("📚 เมนูหลัก")

default_name = st.session_state.get("user_name", "")
user_name = st.sidebar.text_input("👤 Username", value=default_name)
st.session_state.user_name = user_name.strip()
st.session_state.username = st.session_state.user_name # compatibility

st.sidebar.caption(f"ผู้ใช้: {st.session_state.get('user_name','') or '(ไม่ระบุ)'}")
page = st.sidebar.radio("เลือกหน้า", ["Home", "Lessons", "Quiz", "Dashboard"])
history = load_history()
if page == "Home":
    set_app_context(page, st.session_state.get("user_name",""))
    st.title("🐍 Python Learning App ")
    st.write(
        "ฉบับละเอียด: บทเรียนทุกหัวข้อมี Objectives, Key ideas, Examples, Pitfalls, "
        "และแบบฝึก Try it + Quiz เพื่อทบทวนความเข้าใจ"
    )
    st.image("https://static-assets.codecademy.com/assets/course-landing-page/meta/16x9/learn-python-3.jpg", caption="OOP Diagram", use_container_width=True)


elif page == "Lessons":
    st.title("📘 บทเรียน Python ")
    key = st.selectbox("เลือกบทเรียน", list(lessons.keys()), format_func=lambda k: lessons[k]["title"])
    # ส่งเฉพาะส่วนต้น ๆ ของบทเรียนกัน prompt ยาวเกิน
    lesson_excerpt = lessons[key]["content"][:1200]
    set_app_context(page, st.session_state.get("user_name",""), lesson_key=key, extra={"lesson_excerpt": lesson_excerpt})
    st.subheader(lessons[key]["title"])
    st.markdown(lessons[key]["content"])
    # ----- Inline Playground (per-lesson) -----

    st.divider()
    st.markdown("### 🧪 ลองรันโค้ดตัวอย่าง (Inline Playground)")

    # โค้ดตั้งต้น (จะเปลี่ยนให้เหมาะกับบทเรียนก็ได้)
    starter = {
        "print": 'print("Hello from Playground!")',
        "variables": 'name="Alice"\nage=20\nprint(f"{name} is {age}")',
        "datatypes": 'values=[0,1,"", "x",[],[1]]\nprint([bool(v) for v in values])',
        "strings": 's="python"\nprint(s.upper(), s.title(), s[::-1])',
        "lists": 'nums=[1,2,3]\nnums.append(4)\nprint(nums)',
        "tuples": 't=(1,2,3)\na,b,c=t\nprint(a,b,c)',
        "sets": 'print({1,2,2,3} | {3,4})',
        "dictionaries": 'd={"a":1,"b":2}\nprint(d.get("c","N/A"))',
        "if_else": 'x=7\nprint("big" if x>5 else "small")',
        "while_loop": 'i=0\nwhile i<3:\n    print(i)\n    i+=1',
        "for_loop": 'for i in range(3):\n    print(i)',
        "functions": 'def add(a,b):\n    return a+b\nprint(add(3,5))',
        "booleans_operators": 'print(10>5 and 3>1)',
        "casting": 'print(int("123"), float("3.14"), str(100))',
        "classes": 'class Dog:\n    def __init__(self,n): self.n=n\n    def bark(self): print(self.n,"woof")\nDog("Buddy").bark()',
    }

    default_code = starter.get(key, 'print("Ready to run!")')
    code = st.text_area("พิมพ์โค้ด Python ของคุณที่นี่:", value=default_code, height=220)

    col_run, col_reset = st.columns([1,1])

    # สภาพแวดล้อมจำลองแบบเบสิก (จำกัด builtins ระดับหนึ่ง)
    if "lesson_envs" not in st.session_state:
        st.session_state.lesson_envs = {}

    env = st.session_state.lesson_envs.setdefault(key, {"globals": {}, "locals": {}})

    SAFE_BUILTINS = {
        "print": print, "range": range, "len": len, "enumerate": enumerate,
        "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
        "all": all, "any": any, "map": map, "filter": filter, "zip": zip,
        "sorted": sorted
    }
    safe_globals = {"__builtins__": SAFE_BUILTINS}

    with col_run:
        if st.button("▶️ Run code", use_container_width=True):
            buf_out, buf_err = io.StringIO(), io.StringIO()
            try:
                # แยก env ต่อบทเรียน เพื่อให้ตัวแปรในบทนั้น ๆ อยู่ต่อเนื่องได้
                g = env["globals"] or safe_globals.copy()
                l = env["locals"] or {}
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    exec(code, g, l)
                env["globals"], env["locals"] = g, l  # เก็บสถานะต่อรอบ
                out = buf_out.getvalue()
                err = buf_err.getvalue()
                if out:
                    st.success("Standard Output:")
                    st.code(out, language="text")
                if err:
                    st.warning("Standard Error:")
                    st.code(err, language="text")
                if not out and not err:
                    st.info("✓ โค้ดรันสำเร็จ (ไม่มีเอาท์พุต)")
            except Exception:
                st.error("เกิดข้อผิดพลาดระหว่างรันโค้ด:")
                st.code(traceback.format_exc(), language="text")

    with col_reset:
        if st.button("🧹 Reset environment", help="ล้างตัวแปร/สถานะของบทเรียนนี้", use_container_width=True):
            st.session_state.lesson_envs[key] = {"globals": {}, "locals": {}}
            st.success("รีเซ็ตสภาพแวดล้อมเรียบร้อย")

elif page == "Quiz":
    st.title("📝 แบบทดสอบท้ายบท ")
    key = st.selectbox(
        "เลือกบทเรียนสำหรับทำ Quiz",
        list(lessons.keys()),
        format_func=lambda k: lessons[k]["title"]
    )
    set_app_context(page, st.session_state.get("user_name",""), lesson_key=key, extra={"quiz_questions": len(lessons[key].get("quiz", []))})
    questions = lessons[key].get("quiz", [])

    if not questions:
        st.info("บทเรียนนี้ยังไม่มีคำถาม")
    else:
        user_answers = []
        for i, q in enumerate(questions):
            st.write(f"**คำถามที่ {i+1}: {q['question']}**")
            choice = st.radio("เลือกคำตอบ", q["choices"], key=f"{key}_{i}")
            user_answers.append((q, choice))

        if st.button("ส่งคำตอบและบันทึกผล"):
            score = sum(1 for q, c in user_answers if c == q["answer"])
            max_score = len(questions)
            name_for_save = st.session_state.get("user_name", "").strip() or "(ไม่ระบุ)"

            st.success(f"คุณได้ {score} / {max_score} คะแนน 🎉")
            history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "lesson": key,
                "score": score,
                "max_score": max_score,
                "user": name_for_save,
            })
            save_history(history)

elif page == "Dashboard":
    set_app_context(page, st.session_state.get("user_name",""), extra={"records": len(history)})
    st.title("📊 สถิติของคุณ")
    if not history:
        st.info("ยังไม่มีบันทึกผลการทดสอบ")
    else:
        # สร้างตารางเดียวให้จบ
        rows = []
        for h in history:
            lesson_key = h.get("lesson")
            rows.append({
                "วันที่-เวลา": h.get("timestamp"),
                "ผู้ใช้": h.get("user", "(ไม่ระบุ)"),
                "บทเรียน": lessons[lesson_key]["title"] if lesson_key in lessons else str(lesson_key),
                "คะแนน": h.get("score", 0),
                "เต็ม": h.get("max_score", 0),
            })
        df = pd.DataFrame(rows)

        # แปลงเวลา + คิดร้อยละ
        df["วันที่-เวลา"] = pd.to_datetime(df["วันที่-เวลา"], errors="coerce")
        df["ร้อยละ (%)"] = (df["คะแนน"] / df["เต็ม"].replace(0, pd.NA) * 100).astype("float").round(2).fillna(0.0)

        # ตัวกรองตามชื่อผู้ใช้
        names = ["ทั้งหมด"] + sorted(df["ผู้ใช้"].dropna().unique().tolist())
        sel = st.selectbox("กรองตามผู้ใช้", names)
        if sel != "ทั้งหมด":
            df = df[df["ผู้ใช้"] == sel]

        st.dataframe(df.sort_values("วันที่-เวลา", ascending=False)[["วันที่-เวลา", "ผู้ใช้", "บทเรียน", "คะแนน", "เต็ม", "ร้อยละ (%)"]], use_container_width=True)

        st.write("### 📈 สรุปผลรวม (ตามตัวกรอง)")
        st.write(f"- จำนวนครั้งที่ทำแบบทดสอบ: **{len(df)}**")
        st.write(f"- คะแนนเฉลี่ย: **{df['ร้อยละ (%)'].mean():.2f}%**")

        # กราฟเล็ก ๆ (ถ้าข้อมูลพอ)
        if len(df) >= 2 and df["วันที่-เวลา"].notna().any():
            st.write("### แนวโน้มคะแนนตามเวลา")
            trend = df.dropna(subset=["วันที่-เวลา"]).sort_values("วันที่-เวลา")[["วันที่-เวลา", "ร้อยละ (%)"]].set_index("วันที่-เวลา")
            st.line_chart(trend)

        st.write("### คะแนนเฉลี่ยแยกตามบทเรียน")
        by_lesson = df.groupby("บทเรียน", as_index=False)["ร้อยละ (%)"].mean().sort_values("ร้อยละ (%)", ascending=False)
        st.bar_chart(by_lesson.set_index("บทเรียน"))

corner_chat()














