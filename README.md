# roast-my-room

A Go program that looks through your webcam and roasts your room, fully offline.

It takes a photo with [imagesnap](https://github.com/rharder/imagesnap), shows it to a small vision model running inside the Go program with [yzma](https://github.com/hybridgroup/yzma) (Go + [llama.cpp](https://github.com/ggml-org/llama.cpp), no CGo, no server), and reads the roast out loud with the Mac's built-in `say` command. No API key, no Wi-Fi needed once the model is downloaded.

Built for the Build It in Go video "My Go Program Won't Stop Roasting My Room" (video link coming Nov 12). Huge thanks to [Ron Evans](https://github.com/deadprogram) for yzma: go star it.

## Run it (macOS, Apple Silicon)

```sh
./setup.sh        # once: imagesnap, yzma, llama.cpp (Metal), Qwen3-VL 4B
./roast-my-room   # roasts you, again and again, until you press Ctrl-C
```

Every round takes a new photo and says something new: the robot sees its last six roasts and is told not to repeat them, and each round it's nudged toward a different part of the room. The first Ctrl-C lets it have the last word; a second one quits right away.

### Clean-up mode

```sh
./roast-my-room -clean -name Marian
```

The robot gives you one order at a time ("Clean your room, Marian. Start here. Make the bed."), then checks a new photo every 10 seconds. A separate, neutral check compares a photo from when it gave the order with the photo from now (both shrunk to 640 pixels wide, to stay fast), decides whether the order is done, and prints what it decided. Keep the camera still so the two photos line up. Press Enter during the countdown to scan right away. Before giving a new order, it checks the order isn't already done. Nothing changed: it roasts you and repeats the order. Caught you cleaning: "There you go, Marian..." Done: "Finally!" and the next order. After three orders (`-chores 3`), it approves the room and stops.

The model files are about 3 GB and go in `~/models`. The llama.cpp libraries go in `~/yzma/lib`.

## Options

| Flag | What it does |
|---|---|
| `-image room.jpg` | roast a photo instead of taking one |
| `-camera "iPhone Camera"` | pick a webcam (list them with `imagesnap -l`) |
| `-voice Samantha` | pick a `say` voice (list them with `say -v '?'`); Samantha is the default, try Fred or Zarvox for a robot sound |
| `-rate 160` | speaking speed in words per minute |
| `-quiet` | print the roasts without saying them |
| `-clean` | give one cleaning order at a time until the room passes |
| `-name Marian` | what the robot calls you |
| `-first "Make the bed."` | with `-clean`, choose the first order |
| `-chores 3` | with `-clean`, how many orders before the room is approved |
| `-once` | roast one time and stop |
| `-pause 5s` | wait between rounds (default 0, or 10s with `-clean`) |
| `-model`, `-mmproj`, `-lib` | use other model files or a different llama.cpp folder |
| `-v` | show llama.cpp's logs |

The first run asks for camera permission for your terminal app.

## How it works

1. `imagesnap` saves `snapshot.jpg`.
2. yzma loads the model once (the brain) and the projector (the eyes) with llama.cpp on the Metal GPU.
3. The photo and a deadpan "robot room inspector" system prompt go in; three dry sentences come out.
4. `say` reads them, and it starts over with a new photo.

The prompt roasts the room and the things in it, never a person.

## License

MIT
