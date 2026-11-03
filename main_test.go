package main

import (
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"testing"
)

func TestParseClean(t *testing.T) {
	cases := []struct{ in, v, roast, order string }{
		{"VERDICT: NONE\nROAST: The bed has given up. So have you.\nORDER: Make the bed.", "NONE", "The bed has given up. So have you.", "Make the bed."},
		{"**VERDICT:** NOT DONE\n**ROAST:** Still unmade.\n**ORDER:** \"Make the bed.\"", "NOT DONE", "Still unmade.", "Make the bed."},
		{"Verdict: Working\nRoast: Progress detected. Minimal.\nOrder: Make the bed.", "WORKING", "Progress detected. Minimal.", "Make the bed."},
		{"VERDICT: DONE\nROAST: The vanity is next.\nORDER: Put the bottles away.", "DONE", "The vanity is next.", "Put the bottles away."},
		{"Scanning room. A bed. Chaos.", "", "Scanning room. A bed. Chaos.", ""},
		{"- VERDICT: IN PROGRESS\n- ROAST: Slowly.\n- ORDER: Fold it.", "WORKING", "Slowly.", "Fold it."},
	}
	for _, c := range cases {
		v, r, o := parseClean(c.in)
		if v != c.v || r != c.roast || o != c.order {
			t.Errorf("parseClean(%q) = %q, %q, %q; want %q, %q, %q", c.in, v, r, o, c.v, c.roast, c.order)
		}
	}
	if comma("Marian") != ", Marian" || comma("") != "" || prefix("Marian") != "Marian. " {
		t.Error("name helpers")
	}
}

func TestParseJudge(t *testing.T) {
	for in, want := range map[string]string{"YES": "DONE", "Yes.": "DONE", " PARTLY": "WORKING", "Partly, the sheets": "WORKING", "NO": "NOT DONE", "no": "NOT DONE", "": "NOT DONE"} {
		if got := parseJudge(in); got != want {
			t.Errorf("parseJudge(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestShrink(t *testing.T) {
	dir := t.TempDir()
	src, dst := dir+"/big.jpg", dir+"/small.jpg"
	img := image.NewRGBA(image.Rect(0, 0, 1920, 1080))
	for y := 0; y < 1080; y++ {
		for x := 0; x < 1920; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y), 128, 255})
		}
	}
	f, _ := os.Create(src)
	jpeg.Encode(f, img, nil)
	f.Close()

	if err := shrink(src, dst, 640); err != nil {
		t.Fatal(err)
	}
	f, _ = os.Open(dst)
	defer f.Close()
	got, err := jpeg.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	if b := got.Bounds(); b.Dx() != 640 || b.Dy() != 360 {
		t.Errorf("shrunk to %dx%d, want 640x360", b.Dx(), b.Dy())
	}
}

func TestSimilar(t *testing.T) {
	a := "Bed unmade, scattered pillows, cluttered side table with personal items, bookshelf with various books."
	if !similar(a, a) {
		t.Error("a roast should be similar to itself")
	}
	if similar(a, "That chair has given up on life. The lamp is from another decade. Even the dust is bored.") {
		t.Error("different roasts flagged as similar")
	}
	if similar(a, "") {
		t.Error("empty last roast should not be similar")
	}
}
