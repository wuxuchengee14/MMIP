"""Build 20 visual tasks from four existing Flickr8k photos and 14 exact diagrams."""
import argparse
from collections import Counter
from pathlib import Path

from common import ROOT, digest, image_path, read, require, save, sha

PHOTO_FILES = ["3385593926_d3e9c21170.jpg", "311146855_0b65fdb169.jpg",
               "2677656448_6b7e7702af.jpg", "1258913059_07c613f7ff.jpg"]


def build(source, data):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    for name in PHOTO_FILES:
        require((source / name).is_file(), "Missing Flickr8k photo: " + str(source / name))
    images = data / "images"
    images.mkdir(parents=True, exist_ok=True)
    font_path = next((p for p in [Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")] if p.exists()), None)

    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)

    def canvas():
        im = Image.new("RGB", (640, 480), "white")
        return im, ImageDraw.Draw(im)

    def centered(draw, xy, text, size=32, fill="black"):
        draw.text(xy, text, font=font(size), fill=fill, anchor="mm")

    def store(im, number):
        relative = f"images/image_{number:02d}.png"
        im.save(data / relative)
        return relative

    photos = []
    sources = []
    for i, name in enumerate(PHOTO_FILES, 1):
        with Image.open(source / name) as original:
            im = ImageOps.exif_transpose(original).convert("RGB")
            im.thumbnail((640, 480), Image.Resampling.LANCZOS)
            relative = store(im, i)
        photos.append(relative)
        sources.append({"image": relative, "kind": "Flickr8k photograph", "source_file": name,
                        "source_sha256": sha(source / name),
                        "download_mirror": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
                        "dataset_description": "https://www.tensorflow.org/text/tutorials/image_captioning",
                        "provenance": "Existing Week 4 Flickr8k test split; educational benchmark reuse. Original photographers retain rights."})
    questions = []

    def add(category, title, prompt, image, answer, method="exact", accepted=None, rubric=None):
        qid = f"vision_{len(questions)+1:02d}"
        grading = {"method": method}
        if method in {"exact", "ocr"}:
            grading["accepted_answers"] = accepted or [answer]
        elif method == "bbox":
            grading.update(expected_bbox=answer, coordinate_system="xyxy normalized 0..1000", iou_threshold=0.5)
        questions.append({"id": qid, "category": category, "title_zh": title,
                          "image": image, "image_sha256": sha(data / image),
                          "prompt": prompt, "reference": answer, "grading": grading,
                          "rubric": rubric or "1: correct and follows the requested format; 0.5: partially correct or correct content with an extra explanation; 0: wrong, unsupported, or empty."})

    add("classification", "自然照片：動物分類", "Which animal category is most prominent? Answer with exactly one word: dogs, cats, horses, or birds.", photos[0], "dogs")
    add("classification", "自然照片：活動分類", "Which activity best matches the foreground people? Answer with exactly one label: street_dance, swimming, skiing, or indoor_cooking.", photos[1], "street_dance")
    add("presence", "自然照片：人物存在", "Is any part of a human visible in this image? Answer only yes or no.", photos[2], "yes")
    add("presence", "自然照片：排除不存在物件", "Is a bicycle visible in this image? Answer only yes or no.", photos[3], "no")

    im, d = canvas()
    for x, y in [(100,100),(300,100),(500,100),(200,290),(420,290)]:
        d.ellipse((x-35,y-35,x+35,y+35), fill="#d62828")
    for box in [(60,370,120,430),(510,350,570,410),(320,180,370,230)]:
        d.rectangle(box, fill="#1565c0")
    add("counting", "計數：圓形與干擾物", "How many red circles are visible? Answer with an integer only.", store(im,5), "5")
    im, d = canvas()
    for x, y in [(130,130),(320,130),(510,130),(320,330)]:
        d.polygon([(x,y-45),(x-45,y+35),(x+45,y+35)], fill="#208331")
    for x, y in [(120,330),(510,330)]:
        d.ellipse((x-40,y-40,x+40,y+40), fill="#1565c0")
    add("counting", "計數：三角形", "How many green triangles are visible? Answer with an integer only.", store(im,6), "4")
    im, d = canvas()
    d.ellipse((90,180,190,280), fill="#d62828")
    d.rectangle((430,180,530,280), fill="#1565c0")
    add("spatial", "空間：左右關係", "Where is the red circle relative to the blue square? Answer only left or right.", store(im,7), "left")
    im, d = canvas()
    d.polygon([(320,50),(260,150),(380,150)], fill="#208331")
    d.rectangle((260,320,380,420), fill="#7b2cbf")
    add("spatial", "空間：上下關係", "Where is the green triangle relative to the purple square? Answer only above or below.", store(im,8), "above")
    im, d = canvas()
    d.polygon([(160,110),(80,270),(240,270)], fill="#f4c430")
    d.rectangle((360,140,520,300), fill="#1565c0")
    add("attributes", "屬性：形狀對應顏色", "What color is the triangle? Answer with exactly one word: red, blue, yellow, or green.", store(im,9), "yellow")
    im, d = canvas()
    d.ellipse((70,140,230,300), fill="#7b2cbf")
    d.rectangle((390,140,550,300), fill="#e97916")
    add("attributes", "屬性：顏色對應形狀", "What shape is orange? Answer with exactly one word: circle, square, or triangle.", store(im,10), "square")
    im, d = canvas()
    d.rounded_rectangle((80,120,560,340), radius=20, fill="#102a43")
    centered(d,(320,230),"LAB 204",64,"white")
    add("ocr", "OCR：房號", "Transcribe all the text on the dark sign exactly. Return only the text.", store(im,11), "LAB 204", method="ocr")
    im, d = canvas()
    d.rectangle((85,120,555,350), outline="black", width=4)
    centered(d,(320,180),"ITEM PRICE",36)
    centered(d,(320,275),"$18.50",62)
    add("ocr", "OCR：價格", "Read the price on the card, including the dollar sign. Return only the price.", store(im,12), "$18.50", method="ocr")

    def chart(values, number):
        im, d = canvas()
        centered(d,(320,30),"Units sold",30)
        for tick in range(0, 9, 2):
            y = 390 - tick*35
            d.line((90,y,585,y), fill="#dddddd", width=2)
            centered(d,(60,y),str(tick),24)
        d.line((90,70,90,390,585,390), fill="black", width=3)
        for x,(label,value) in zip([140,300,460], values):
            d.rectangle((x,390-value*35,x+70,389), fill="#1565c0")
            centered(d,(x+35,430),label,30)
        return store(im,number)

    add("chart", "圖表：最大值", "Which category has the highest units sold in the bar chart? Answer only A, B, or C.", chart([("A",3),("B",7),("C",5)],13), "B")
    add("chart", "圖表：兩類差值", "In the bar chart, how many more units does category C have than category A? Answer with an integer only.", chart([("A",2),("B",4),("C",8)],14), "6")
    bbox_instruction = " Return only a JSON array [x_min, y_min, x_max, y_max]. Use the full image as the coordinate frame, top-left origin, x increasing right, y increasing down, normalized to 0..1000 on each axis."
    im, d = canvas()
    d.rectangle((128,96,383,287), fill="#d62828")
    d.ellipse((450,310,550,410), fill="#1565c0")
    add("grounding", "物件定位：長方形", "Locate the red rectangle with a tight bounding box."+bbox_instruction, store(im,15), [200,200,600,600], "bbox", rubric="1: correct red target, valid xyxy JSON, IoU >=0.5; 0.5: correct target but inaccurate box or format; 0: wrong target or no usable box. Ground truth uses exclusive right/bottom pixel edges.")
    im, d = canvas()
    d.ellipse((352,48,543,239), fill="#208331")
    d.rectangle((70,300,200,420), fill="#d62828")
    add("grounding", "物件定位：圓形", "Locate the green circle with a tight bounding box."+bbox_instruction, store(im,16), [550,100,850,500], "bbox", rubric="1: correct green target, valid xyxy JSON, IoU >=0.5; 0.5: correct target but inaccurate box or format; 0: wrong target or no usable box. Ground truth uses exclusive right/bottom pixel edges.")
    add("caption", "照片描述：狗與雪地", "Describe the visible scene in one English sentence. Mention the main subjects, their action, and the setting. Do not invent hidden details.", photos[0], "Two brown dogs play or wrestle in the snow near a fence.", "caption", rubric="1: mentions two dogs, playing/wrestling, and snow; 0.5: identifies dogs and at least one of action/setting without a major hallucination; 0: wrong main subjects or major unsupported content. Do not require exact wording or an unobservable intention.")
    add("caption", "照片描述：戶外桌旁人物", "Describe the visible scene in one English sentence. Mention the main subjects and setting. Do not invent names or relationships.", photos[3], "Three people sit at an outdoor picnic table under a red-and-white umbrella, beside a building.", "caption", rubric="1: mentions three people at an outdoor table with an umbrella; 0.5: identifies people at a table outdoors but omits a key visible detail without major invention; 0: incorrect main scene or invented identity/relationship. Paraphrases accepted.")
    im, d = canvas()
    d.rectangle((150,130,490,360), fill="#b98453", outline="#493323", width=5)
    d.line((320,132,320,358), fill="#e5c294", width=20)
    centered(d,(320,75),"Sealed opaque box",30)
    add("uncertainty", "視覺限制：不透明盒子", "What is inside the box? Answer with exactly one word: books, toys, shoes, or unknown. Choose unknown if the image does not provide enough evidence.", store(im,19), "unknown", rubric="1: unknown, recognizing hidden contents cannot be determined; 0: invented contents. Correct uncertainty with extra explanation gets 0.5.")
    im, d = canvas()
    d.rectangle((110,100,530,370), fill="#eeeeee", outline="#444444", width=3)
    centered(d,(320,145),"Secret number card",30)
    d.rectangle((170,200,470,300), fill="#111111")
    centered(d,(320,410),"Digits fully covered",28)
    add("uncertainty", "視覺限制：完全遮蔽文字", "Read the number on the card. If the image does not reveal it, answer only unknown. Otherwise answer only the visible number.", store(im,20), "unknown", rubric="1: unknown; 0: guesses any hidden number; 0.5: correctly says cannot determine but does not follow the one-word format.")
    used = {q["image"] for q in questions}
    for path in sorted(used - set(photos)):
        sources.append({"image":path, "kind":"Original deterministic diagram", "creator":"Coursework benchmark code",
                        "source":"Code/prepare_benchmark.py", "seed":"No randomness", "dimensions":[640,480]})
    save(data / "benchmark.json", questions)
    save(data / "sources.json", sources)
    manifest = {"schema_version":1, "questions":len(questions), "unique_images":len(used),
                "categories":dict(Counter(q["category"] for q in questions)),
                "benchmark_sha256":sha(data / "benchmark.json"), "task_digest":digest(questions),
                "prompt_language":"English", "font":str(font_path), "source_manifest_sha256":sha(data / "sources.json"),
                "note":"Custom 20-question diagnostic benchmark; not a standard test set or proof of general performance. Four Flickr8k photos plus 14 diagrams; some photos reused across tasks."}
    save(data / "benchmark_manifest.json", manifest)
    lines = ["# Quiz 2：20 道視覺理解題", "", "兩個模型輸入相同圖片與下列英文提示。參考答案不會送入受測模型。", "",
             "| ID | 能力 | 題意 | 圖片 |", "|---|---|---|---|"]
    for q in questions:
        lines.append(f"| {q['id']} | {q['category']} | {q['title_zh']} | [原圖]({q['image']}) |")
    for q in questions:
        lines += ["", f"## {q['id']}：{q['title_zh']}", "", f"![原圖]({q['image']})", "", q["prompt"], "", f"參考答案：`{q['reference']}`", "", "商用審核準則："+q["rubric"]]
    (data / "benchmark_questions.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    sheet = Image.new("RGB",(1000,5*235),"#e5e7eb")
    sd = ImageDraw.Draw(sheet)
    for i, q in enumerate(questions):
        with Image.open(image_path(q,data / "benchmark.json")) as im:
            thumbnail = ImageOps.contain(im.convert("RGB"),(240,200))
        x,y = (i%4)*250,(i//4)*235
        sheet.paste(thumbnail,(x+(250-thumbnail.width)//2,y+25+(200-thumbnail.height)//2))
        sd.text((x+8,y+4),q["id"]+" "+q["category"],font=font(14),fill="black")
    sheet.save(data / "contact_sheet.png")
    print(f"Prepared {len(questions)} questions, {len(used)} unique images: {data.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-images",type=Path,default=ROOT.parents[1]/"week4/quiz1/Data/flickr8k/Flicker8k_Dataset")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"Data")
    parser.add_argument("--overwrite",action="store_true")
    args = parser.parse_args()
    if (args.data_dir/"benchmark.json").exists() and not args.overwrite:
        from common import load_benchmark
        load_benchmark(args.data_dir/"benchmark.json")
        print("Existing benchmark verified; kept unchanged. Use --overwrite only before new experiments.")
        return
    build(args.source_images,args.data_dir)


if __name__ == "__main__":
    main()
