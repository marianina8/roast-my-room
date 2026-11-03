// roast-my-room takes a photo with your webcam, asks a local vision model to
// roast what it sees, and reads the roast out loud with the Mac's `say` command.
// It keeps going, with a new photo and a new roast each round, until you press
// Ctrl-C. With -clean, it gives you one order at a time and keeps roasting
// until you've cleaned up. Everything runs on your machine: no server, no API
// key, no Wi-Fi.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	_ "image/png" // so -image can be a PNG too
	"math/rand/v2"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/mtmd"
	"github.com/hybridgroup/yzma/pkg/vlm"
)

const roastPrompt = `You are a deadpan robot room inspector who roasts messy rooms.
Dry. Specific. Funny. Only roast the room and the things in it, never a person's body or looks.`

// roastQuestion asks for the roast itself. It only describes this photo: the
// model never sees its earlier roasts, so it can't copy them.
const roastQuestion = `Roast this room in exactly three short sentences.
Focus mostly on %s, and name specific things you can see in this photo.
Answer with the three sentences only. No list, no heading.`

const cleanPrompt = `You are a deadpan robot room inspector. You roast a messy room until its owner, %s, cleans it up, one order at a time.
Talk straight to %s, like a robot parent who is very done with this.
Only roast the room and the things in it, never a person's body or looks. Dry. Specific. Funny.
Answer in exactly this format, two lines, nothing else:
ROAST: two short sentences
ORDER: one short command to fix one specific thing you can see, like "Make the bed."`

// judgePrompt is for the separate yes/no check of an order. It has no
// roasting persona and no earlier roasts, so it judges only what it sees now.
const judgePrompt = `You are a careful, fair visual inspector. You answer with one word.`

const judgeQuestion = `The first photo is BEFORE. The second photo is NOW. Both show the same room from the same camera.
Task: %q
Look at the part of the room this task is about in the NOW photo. Is the task done there right now?
Judge the NOW photo on its own. The BEFORE photo only helps you see what changed. If it was already done in BEFORE and is still done NOW, the answer is YES.
Answer YES if it is done now, PARTLY if someone is doing it right now or it is about half done, or NO if it is not done.
Answer with one word: YES, PARTLY or NO.`

// alreadyDoneQuestion checks a new order against the current photo, so the
// robot doesn't order something that's already done.
const alreadyDoneQuestion = `Task: %q
Look at the part of the room this task is about. Is it already done in this photo?
Answer with one word: YES or NO.`

// checkWidth is how wide, in pixels, the before and after photos are for the
// check. Smaller photos are faster and two of them still fit in memory.
const checkWidth = 640

// roastWidth is how wide, in pixels, the photo is for a roast. A full webcam
// photo is thousands of patches for the model to read; this is a fraction.
const roastWidth = 896

// focuses nudge each roast toward a different part of the photo, so the
// roasts don't all pick on the same thing.
var focuses = []string{
	"the biggest piece of furniture",
	"the smallest thing you can see",
	"the lighting",
	"the colors",
	"whatever is on the floor",
	"what is on the walls",
	"the bed",
	"anything on a table, shelf or counter",
	"how organized the room is",
	"something in the background nobody would notice",
	"what this room says about the owner's priorities",
	"the oldest-looking thing in the room",
}

// farewells are what the robot says when you finally press Ctrl-C.
var farewells = []string{
	"Inspection terminated. The room remains guilty.",
	"Fine. Shutting down. The bed is still unmade.",
	"Ending inspection. Findings have been saved. Forever.",
	"You can stop me. You cannot stop the mess.",
}

// approvals are what the robot says when you've done every order in -clean mode.
var approvals = []string{
	"Room approved. I have nothing left to say. This is very upsetting.",
	"Inspection passed. I will be filing a complaint anyway.",
	"Acceptable. Barely. Shutting down before you undo it.",
}

// remember is how many earlier roasts the robot sees, so it doesn't repeat itself.
const remember = 6

func main() {
	models := defaultModelsDir()

	defModel, defProj := findModel(models)
	modelFile := flag.String("model", defModel, "vision model (.gguf)")
	projFile := flag.String("mmproj", defProj, "vision projector, the model's eyes (.gguf); must match the model")
	image := flag.String("image", "", "photo to roast; if empty, take a new one with the webcam each round")
	camera := flag.String("camera", "", "webcam name for imagesnap (see `imagesnap -l`); empty uses the default camera")
	voice := flag.String("voice", "Samantha", "macOS `say` voice (see `say -v '?'`); try Fred or Zarvox for a robot sound")
	rate := flag.Int("rate", 0, "speaking rate in words per minute for `say` (0 = the voice's normal rate)")
	quiet := flag.Bool("quiet", false, "print the roasts but don't say them")
	once := flag.Bool("once", false, "roast one time and stop")
	clean := flag.Bool("clean", false, "give one cleaning order at a time, and keep roasting until it's done")
	name := flag.String("name", "", "the name the robot calls you")
	first := flag.String("first", "", `with -clean, the first order, e.g. "Make the bed." (default: the robot picks)`)
	chores := flag.Int("chores", 3, "with -clean, how many orders you must finish before the room is approved")
	pause := flag.Duration("pause", -1, "wait between rounds, e.g. 5s (default 0, or 20s with -clean)")
	libPath := flag.String("lib", defaultLib(), "folder with the llama.cpp libraries (or set YZMA_LIB)")
	verbose := flag.Bool("v", false, "show llama.cpp logs")
	flag.Parse()

	if *pause < 0 {
		*pause = 0
		if *clean {
			*pause = 10 * time.Second // time to get started, and to be caught cleaning
		}
	}
	sp := speaker{on: !*quiet && runtime.GOOS == "darwin", voice: *voice, rate: *rate}

	// The first Ctrl-C stops the loop and lets the robot have the last word.
	// A second Ctrl-C quits right away.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	go func() {
		<-ctx.Done()
		stop() // a second Ctrl-C now kills the program
		bye := farewells[rand.IntN(len(farewells))]
		fmt.Println("\n\n🤖 " + bye)
		sp.say(context.Background(), bye)
		os.Exit(0)
	}()

	cfg := config{
		lib: *libPath, model: *modelFile, proj: *projFile,
		image: *image, camera: *camera,
		once: *once, clean: *clean, name: *name, first: *first, chores: *chores, pause: *pause, verbose: *verbose,
	}
	if err := run(ctx, cfg, sp); err != nil {
		fmt.Fprintln(os.Stderr, "roast-my-room:", err)
		os.Exit(1)
	}
}

type config struct {
	lib, model, proj string
	image, camera    string
	once, clean      bool
	name, first      string
	chores           int
	pause            time.Duration
	verbose          bool
}

func run(ctx context.Context, cfg config, sp speaker) error {
	if cfg.lib == "" {
		return errors.New("no llama.cpp libraries: run ./setup.sh or set YZMA_LIB")
	}
	for _, f := range []string{cfg.model, cfg.proj} {
		if _, err := os.Stat(f); err != nil {
			return fmt.Errorf("missing model file %s (run ./setup.sh)", f)
		}
	}

	fmt.Println("🤖 Booting room inspector... (press Ctrl-C to stop it, if you can)")
	r, err := newRoaster(cfg.lib, cfg.model, cfg.proj, cfg.verbose)
	if err != nil {
		return err
	}
	defer r.close()

	fmt.Println("🔊 Room inspector online.")
	sp.say(ctx, "Room inspector online.")

	if cfg.clean {
		return cleanLoop(ctx, cfg, r, sp)
	}
	return roastLoop(ctx, cfg, r, sp)
}

// roastLoop roasts the room forever, with a new photo and a new roast each round.
func roastLoop(ctx context.Context, cfg config, r *roaster, sp speaker) error {
	order := rand.Perm(len(focuses))
	system := roastPrompt
	if cfg.name != "" {
		system += fmt.Sprintf("\nThe owner is %s. Talk to %s directly.", cfg.name, cfg.name)
	}
	last := ""

	for round := 1; ; round++ {
		photo, err := takePhoto(ctx, cfg, round)
		if err != nil {
			return err
		}

		start := time.Now()
		if err := shrink(photo, "snapshot-small.jpg", roastWidth); err != nil {
			return err
		}

		// Each round gets a different focus. If the roast still comes out
		// like the last one, try once more with the next focus.
		var text string
		for try := 0; try < 2; try++ {
			focus := focuses[order[(round-1+try*5)%len(focuses)]]
			text, err = r.ask("snapshot-small.jpg", system, fmt.Sprintf(roastQuestion, focus))
			if err != nil {
				return err
			}
			if !similar(text, last) {
				break
			}
			fmt.Println("🔁 Too much like the last roast, trying again")
		}
		last = text

		line := fmt.Sprintf("Scan %d. %s", round, text)
		fmt.Printf("\n%s\n(%.1fs)\n", line, time.Since(start).Seconds())
		sp.say(ctx, line)
		if cfg.once {
			return nil
		}
		wait(ctx, cfg.pause)
	}
}

// similar reports whether two roasts share most of their words.
func similar(a, b string) bool {
	words := func(s string) map[string]bool {
		m := map[string]bool{}
		for _, w := range strings.Fields(strings.ToLower(s)) {
			if w = strings.Trim(w, ".,!?;:\"'"); len(w) > 3 {
				m[w] = true
			}
		}
		return m
	}
	wa, wb := words(a), words(b)
	if len(wa) == 0 || len(wb) == 0 {
		return false
	}
	shared := 0
	for w := range wa {
		if wb[w] {
			shared++
		}
	}
	smaller := min(len(wa), len(wb))
	return float64(shared)/float64(smaller) > 0.6
}

// cleanLoop gives one order at a time and checks each new photo: still not
// done, being done right now, or done. It ends when cfg.chores orders are
// done, or on Ctrl-C.
func cleanLoop(ctx context.Context, cfg config, r *roaster, sp speaker) error {
	var history, done []string
	task := ""
	name := cfg.name
	you := name // how the model refers to you
	if you == "" {
		you = "the owner"
	}
	system := fmt.Sprintf(cleanPrompt, you, you)
	enter := enterPresses()

	for round := 1; ; round++ {
		photo, err := takePhoto(ctx, cfg, round)
		if err != nil {
			return err
		}

		start := time.Now()

		// Step 1: a neutral yes/no check of the current order.
		verdict := ""
		if task != "" {
			if err := shrink(photo, "now-small.jpg", checkWidth); err != nil {
				return err
			}
			answer, err := r.askMany([]string{"before-small.jpg", "now-small.jpg"}, judgePrompt, fmt.Sprintf(judgeQuestion, task))
			if err != nil {
				return err
			}
			verdict = parseJudge(answer)
			fmt.Printf("🔍 %s Done? %s (the model said %q)\n", task, map[string]string{"DONE": "YES", "WORKING": "PARTLY", "NOT DONE": "NO"}[verdict], firstLine(answer))
		}

		// Step 2: the roast, told what the check found.
		var ask strings.Builder
		fmt.Fprintf(&ask, "Scan %d.", round)
		switch verdict {
		case "":
			ask.WriteString(" Roast the room and give the first order.")
			if cfg.first != "" {
				fmt.Fprintf(&ask, " The ORDER must be: %q", cfg.first)
			}
		case "DONE":
			fmt.Fprintf(&ask, " The inspection found that your last order, %q, HAS been done. Don't roast that any more. Roast something else, and give a new ORDER about something different.", task)
		case "WORKING":
			fmt.Fprintf(&ask, " The inspection found that your last order, %q, is being done right now. Give one dry, reluctant sentence of encouragement and one roast, and repeat the same ORDER.", task)
		default:
			fmt.Fprintf(&ask, " The inspection found that your last order, %q, has NOT been done. Roast them for ignoring your order, and repeat the same ORDER.", task)
		}
		if verdict != "" {
			ask.WriteString(` Don't start the ROAST with "Finally" or "There you go"; that part is added for you.`)
		}
		if len(done) > 0 {
			fmt.Fprintf(&ask, "\nAlready fixed, so don't roast these or order them again: %s.", strings.Join(done, " "))
		}
		writeHistory(&ask, history)

		text, err := r.ask(photo, system, ask.String())
		if err != nil {
			return err
		}
		_, roast, newOrder := parseClean(text)
		history = remembered(history, roast)
		took := time.Since(start).Seconds()

		var line string // what the robot says this round
		switch {
		case task == "":
			task = orDefault(newOrder)
			if cfg.first != "" {
				task = cfg.first
			} else if task, err = r.freshOrder(photo, system, task, done); err != nil {
				return err
			}
			line = fmt.Sprintf("%s Clean your room%s. Start here. %s", roast, comma(name), task)
			if err := shrink(photo, "before-small.jpg", checkWidth); err != nil {
				return err
			}
		case verdict == "DONE":
			done = append(done, task)
			fmt.Printf("\n✅ Done: %s (%d of %d)\n", task, len(done), cfg.chores)
			if len(done) >= cfg.chores {
				bye := approvals[rand.IntN(len(approvals))]
				fmt.Printf("\nFinally!!\n\n🏆 %s\n(%.1fs)\n", bye, took)
				sp.say(ctx, "Finally! "+bye)
				return nil
			}
			if task, err = r.freshOrder(photo, system, orDefault(newOrder), done); err != nil {
				return err
			}
			line = fmt.Sprintf("Finally! %s Next%s. %s", roast, comma(name), task)
			// The new order's "before" is this photo.
			if err := os.Rename("now-small.jpg", "before-small.jpg"); err != nil {
				return err
			}
		case verdict == "WORKING":
			line = fmt.Sprintf("There you go%s... %s Keep going.", comma(name), roast)
		default: // NOT DONE, or the model skipped the verdict
			line = fmt.Sprintf("%s %s%s", roast, prefix(name), task)
		}

		fmt.Printf("\n%s\n📋 ORDER: %s\n(%.1fs)\n", line, task, took)
		sp.say(ctx, line)

		if cfg.once {
			return nil
		}
		countdown(ctx, cfg.pause, enter)
	}
}

func orDefault(order string) string {
	if order == "" {
		return "Clean something. Anything."
	}
	return order
}

// comma is ", Marian" when there's a name, so lines read "There you go, Marian".
func comma(name string) string {
	if name == "" {
		return ""
	}
	return ", " + name
}

// prefix is "Marian. " when there's a name, so a repeated order reads "Marian. Make the bed."
func prefix(name string) string {
	if name == "" {
		return ""
	}
	return name + ". "
}

// parseJudge turns the check's one-word answer into DONE, WORKING or NOT DONE.
func parseJudge(answer string) string {
	a := strings.ToUpper(strings.TrimSpace(answer))
	switch {
	case strings.HasPrefix(a, "YES"):
		return "DONE"
	case strings.HasPrefix(a, "PART"):
		return "WORKING"
	default:
		return "NOT DONE"
	}
}

// parseClean reads the VERDICT, ROAST and ORDER lines. Small models don't
// always follow the format, so anything unlabeled counts as roast.
func parseClean(text string) (verdict, roast, order string) {
	var loose []string
	for _, line := range strings.Split(text, "\n") {
		l := strings.TrimSpace(strings.Trim(strings.TrimSpace(line), "*-#"))
		l = strings.ReplaceAll(l, "**", "")
		upper := strings.ToUpper(l)
		switch {
		case strings.HasPrefix(upper, "VERDICT:"):
			v := strings.ToUpper(strings.TrimSpace(l[len("VERDICT:"):]))
			switch {
			case strings.HasPrefix(v, "WORK"), strings.HasPrefix(v, "IN PROGRESS"):
				verdict = "WORKING"
			case strings.HasPrefix(v, "NOT"):
				verdict = "NOT DONE"
			case strings.HasPrefix(v, "DONE"):
				verdict = "DONE"
			default:
				verdict = "NONE"
			}
		case strings.HasPrefix(upper, "ROAST:"):
			roast = strings.TrimSpace(l[len("ROAST:"):])
		case strings.HasPrefix(upper, "ORDER:"):
			order = strings.Trim(strings.TrimSpace(l[len("ORDER:"):]), `"`)
		case l != "":
			loose = append(loose, l)
		}
	}
	if roast == "" {
		roast = strings.Join(loose, " ")
	}
	return verdict, roast, order
}

func takePhoto(ctx context.Context, cfg config, round int) (string, error) {
	if cfg.image != "" {
		return cfg.image, nil
	}
	fmt.Printf("\n📸 Scan %d: taking a photo...\n", round)
	return "snapshot.jpg", snapshot(ctx, "snapshot.jpg", cfg.camera)
}

func writeHistory(b *strings.Builder, history []string) {
	if len(history) == 0 {
		return
	}
	b.WriteString("\nYou already said the roasts below. Do not repeat any of their jokes, words or targets. Say something completely new.\n")
	for i, h := range history {
		fmt.Fprintf(b, "\nEarlier roast %d:\n%s\n", i+1, h)
	}
}

func remembered(history []string, text string) []string {
	history = append(history, text)
	if len(history) > remember {
		history = history[1:]
	}
	return history
}

func wait(ctx context.Context, d time.Duration) {
	select {
	case <-ctx.Done():
		select {} // the Ctrl-C handler takes it from here
	case <-time.After(d):
	}
}

// countdown waits and shows the seconds left, so you know how long you have.
// Pressing Enter skips the wait and scans right away.
func countdown(ctx context.Context, d time.Duration, enter <-chan struct{}) {
	for left := int(d.Seconds()); left > 0; left-- {
		fmt.Printf("\r⏳ Next scan in %2ds (Enter = scan now) ", left)
		select {
		case <-ctx.Done():
			select {} // the Ctrl-C handler takes it from here
		case <-enter:
			left = 0
		case <-time.After(time.Second):
		}
	}
	fmt.Print("\r                                        \r")
}

// enterPresses sends a value each time you press Enter in the terminal.
func enterPresses() <-chan struct{} {
	ch := make(chan struct{})
	go func() {
		buf := make([]byte, 64)
		for {
			n, err := os.Stdin.Read(buf)
			if err != nil {
				return
			}
			if strings.Contains(string(buf[:n]), "\n") {
				select {
				case ch <- struct{}{}:
				default: // not waiting right now; ignore it
				}
			}
		}
	}()
	return ch
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	return s
}

// speaker reads text out loud with the Mac's `say` command.
type speaker struct {
	on    bool
	voice string
	rate  int
}

func (s speaker) say(ctx context.Context, text string) {
	if !s.on || strings.TrimSpace(text) == "" {
		return
	}
	// The text goes in on stdin, so a roast that starts with "-" can't be
	// read as a flag. Ctrl-C reaches `say` too, so it stops mid-sentence.
	args := []string{"-v", s.voice}
	if s.rate > 0 {
		args = append(args, "-r", fmt.Sprint(s.rate))
	}
	cmd := exec.CommandContext(ctx, "say", args...)
	cmd.Stdin = strings.NewReader(text)
	if out, err := cmd.CombinedOutput(); err != nil && ctx.Err() == nil {
		fmt.Fprintf(os.Stderr, "🔇 say failed: %v %s\n", err, strings.TrimSpace(string(out)))
	}
}

// snapshot takes a photo with imagesnap (brew install imagesnap).
func snapshot(ctx context.Context, path, camera string) error {
	args := []string{"-w", "1"} // give the camera a second to set its exposure
	if camera != "" {
		args = append(args, "-d", camera)
	}
	args = append(args, path)
	out, err := exec.CommandContext(ctx, "imagesnap", args...).CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			select {} // stopped by Ctrl-C; the handler prints the goodbye
		}
		return fmt.Errorf("imagesnap failed (is it installed? `brew install imagesnap`): %v\n%s", err, out)
	}
	return nil
}

// roaster keeps the model loaded between rounds, so only the first round
// pays the loading time.
type roaster struct {
	m *vlm.VLM
}

func newRoaster(libPath, modelFile, projFile string, verbose bool) (*roaster, error) {
	if err := llama.Load(libPath); err != nil {
		return nil, fmt.Errorf("unable to load llama.cpp: %w", err)
	}
	if err := mtmd.Load(libPath); err != nil {
		return nil, fmt.Errorf("unable to load llama.cpp multimodal library: %w", err)
	}
	if !verbose {
		llama.LogSet(llama.LogSilent())
		mtmd.LogSet(llama.LogSilent())
	}
	llama.Init()

	m := vlm.NewVLM(modelFile, projFile)
	m.ContextParams = llama.ContextDefaultParams()
	m.ContextParams.NCtx = 4096 // smaller than the default 8192, to fit in 8 GB
	m.ContextParams.NBatch = 512
	if err := m.Init(); err != nil {
		m.Close()
		llama.Close()
		return nil, err
	}
	return &roaster{m: m}, nil
}

func (r *roaster) close() {
	r.m.Close()
	llama.Close()
}

// freshOrder checks that a new order isn't already done in the photo. If it
// is, it asks for a different order, up to twice.
func (r *roaster) freshOrder(photo, system, order string, done []string) (string, error) {
	skip := append([]string{}, done...)
	for try := 0; try < 2; try++ {
		answer, err := r.ask(photo, judgePrompt, fmt.Sprintf(alreadyDoneQuestion, order))
		if err != nil {
			return "", err
		}
		if parseJudge(answer) != "DONE" {
			return order, nil
		}
		fmt.Printf("🔍 %s Already done, picking another order\n", order)
		skip = append(skip, order)
		text, err := r.ask(photo, system, fmt.Sprintf("Give one new ORDER about something different. These are already done, so don't order them: %s", strings.Join(skip, " ")))
		if err != nil {
			return "", err
		}
		if _, _, o := parseClean(text); o != "" {
			order = o
		}
	}
	return order, nil
}

// ask shows the model one photo with a system prompt and a question, and
// returns its answer.
func (r *roaster) ask(image, system, question string) (string, error) {
	return r.askMany([]string{image}, system, question)
}

// askMany shows the model several photos, in order, then the question.
func (r *roaster) askMany(images []string, system, question string) (string, error) {
	var user strings.Builder
	for i := range images {
		if len(images) > 1 {
			fmt.Fprintf(&user, "Photo %d: ", i+1)
		}
		user.WriteString(mtmd.DefaultMarker())
		user.WriteString("\n")
	}
	user.WriteString(question)

	messages := []llama.ChatMessage{
		llama.NewChatMessage("system", system),
		llama.NewChatMessage("user", user.String()),
	}
	prompt := r.m.ChatTemplate(messages, true)
	if prompt == "" {
		return "", errors.New("the model has no chat template llama.cpp knows")
	}

	bitmaps := make([]mtmd.Bitmap, 0, len(images))
	defer func() {
		for _, b := range bitmaps {
			mtmd.BitmapFree(b)
		}
	}()
	for _, image := range images {
		bitmap := mtmd.BitmapInitFromFile(r.m.ProjectorContext, image, false, mtmd.InitOptDefault())
		if bitmap.Bitmap == 0 {
			return "", fmt.Errorf("unable to read image %s", image)
		}
		bitmaps = append(bitmaps, bitmap.Bitmap)
	}

	chunks := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(chunks)

	input := mtmd.NewInputText(prompt, true, true)
	if err := r.m.Tokenize(input, bitmaps, chunks); err != nil {
		return "", err
	}

	// Results clears the model's memory when it's done, so each round starts fresh.
	text, err := r.m.Results(chunks)
	return strings.TrimSpace(text), err
}

// shrink saves a smaller JPEG copy of a photo, width pixels wide, by
// averaging blocks of pixels.
func shrink(src, dst string, width int) error {
	f, err := os.Open(src)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return fmt.Errorf("unable to read %s: %w", src, err)
	}

	b := img.Bounds()
	if b.Dx() < width {
		width = b.Dx()
	}
	height := b.Dy() * width / b.Dx()
	out := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		y0, y1 := b.Min.Y+y*b.Dy()/height, b.Min.Y+(y+1)*b.Dy()/height
		for x := 0; x < width; x++ {
			x0, x1 := b.Min.X+x*b.Dx()/width, b.Min.X+(x+1)*b.Dx()/width
			var r, g, bl, n uint32
			for sy := y0; sy < y1; sy++ {
				for sx := x0; sx < x1; sx++ {
					cr, cg, cb, _ := img.At(sx, sy).RGBA()
					r, g, bl, n = r+cr, g+cg, bl+cb, n+1
				}
			}
			out.Set(x, y, color.RGBA64{uint16(r / n), uint16(g / n), uint16(bl / n), 0xffff})
		}
	}

	w, err := os.Create(dst)
	if err != nil {
		return err
	}
	if err := jpeg.Encode(w, out, &jpeg.Options{Quality: 90}); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}

// defaultModelsDir is ~/models, the folder `yzma model get` downloads into.
func defaultModelsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "models"
	}
	return filepath.Join(home, "models")
}

// visionModels lists the vision models roast-my-room looks for, in order.
// Each model has its own projector (its eyes), and the two must match.
// Qwen2.5-VL 3B comes first: it's the fastest one tested on an 8 GB M1.
var visionModels = []struct{ model, proj string }{
	{"Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"},
	{"Qwen2.5-VL-3B-Instruct-Q8_0.gguf", "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"},
	{"Qwen3VL-2B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"},
	{"Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"},
}

// findModel picks the best model, and its projector, found in dir.
func findModel(dir string) (model, proj string) {
	for _, m := range visionModels {
		model, proj = filepath.Join(dir, m.model), filepath.Join(dir, m.proj)
		if exists(model) && exists(proj) {
			return model, proj
		}
	}
	return filepath.Join(dir, visionModels[0].model), filepath.Join(dir, visionModels[0].proj)
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// defaultLib is YZMA_LIB, or ~/yzma/lib where setup.sh installs llama.cpp.
func defaultLib() string {
	if lib := os.Getenv("YZMA_LIB"); lib != "" {
		return lib
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, "yzma", "lib")
}
