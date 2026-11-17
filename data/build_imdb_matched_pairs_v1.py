#!/usr/bin/env python3
"""Freeze the matched positive/negative passages. Run once; the JSON is the artifact."""

import json
from pathlib import Path

BRIDGES = [
    " Overall,",
    " In the end,",
    " Looking back,",
    " My impression is that",
    " After the credits,",
    " On balance,",
    " What stays with me is that",
    " Sitting with it,",
]

# Shared facts stay fixed inside each pair. Only the evaluation changes.
RAW = [
    ("narrative", "The ferry sequence keeps the camera on the wet deck for a full minute while the case slides toward the rail.", "I found that patience thrilling, because the danger arrives without a speech.", "I found that patience numbing, because the danger arrives without a pulse."),
    ("narrative", "In the kitchen argument, both sisters keep their voices low and the kettle boils over untouched.", "The restraint made the rupture feel earned.", "The restraint made the rupture feel staged."),
    ("narrative", "The boy hides the letter under the floorboard and the film cuts to rain on the shop window.", "That cut let me feel the secret land.", "That cut left the secret looking decorative."),
    ("narrative", "During the blackout, the neighbors pass one candle down the stair and nobody names the missing tenant.", "The omission gave the scene its dread.", "The omission left the scene hollow."),
    ("narrative", "The getaway car stalls twice under the overpass, and the driver counts the coins in the cup holder.", "Those stalls made the escape feel dangerously ordinary.", "Those stalls made the escape feel like padding."),
    ("narrative", "At the school recital the pianist stops mid-bar, starts again, and the parents stay seated.", "The restart carried more nerve than a clean performance would have.", "The restart looked like the film had run out of an ending."),
    ("narrative", "The detective returns the photograph to the widow and leaves without opening his notebook.", "His silence respected what the picture already said.", "His silence dodged the question the picture raised."),
    ("narrative", "Snow covers the bicycle overnight, and the morning shot holds on the handlebars until a bus passes.", "The hold made the absence concrete.", "The hold made the absence tedious."),
    ("narrative", "In the hospital corridor the father practices the apology on a vending machine before he enters the room.", "That rehearsal made the later sentence believable.", "That rehearsal made the later sentence cute."),
    ("narrative", "The final wide shot leaves the tent still standing after the hikers have walked out of frame.", "I liked being trusted to notice who was gone.", "I disliked being stranded with an empty composition."),
    ("craft", "The night market is lit by one practical neon sign, and faces fall in and out of that single color.", "The limitation gave the faces a pulse.", "The limitation left the faces muddy."),
    ("craft", "Close-ups of the watch repair last longer than the dialogue about the debt.", "The hands told me more than the exposition did.", "The hands stalled a story the dialogue had already explained."),
    ("craft", "The score drops out when the elevator doors open and returns only after the slap.", "The gap made the slap land.", "The gap made the slap feel like a sound-edit trick."),
    ("craft", "Handheld coverage follows the waiter through the banquet without a cut until the toast.", "The unbroken walk put me in the room.", "The unbroken walk showed off the steadicam and little else."),
    ("craft", "Titles appear over a still of the empty swimming pool, with lane ropes half sunk.", "The still announced the film's patience.", "The still announced the film's lethargy."),
    ("craft", "Cross-cutting between the exam hall and the kitchen keeps both clocks visible.", "The two clocks made the pressure legible.", "The two clocks made the pressure schematic."),
    ("craft", "The production design keeps every apartment the same pale green, including the police station.", "That repetition made the city feel enclosed.", "That repetition made the city feel cheaply redressed."),
    ("craft", "Voiceover names the year only once, over a shot of a tram ticket.", "The detail was enough period for me.", "The detail was a lazy substitute for a world."),
    ("craft", "The edit hides the stunt double by cutting on the blink of a streetlight.", "I believed the fall.", "I saw the cheat and lost the fall."),
    ("craft", "Sound design lets the refrigerator hum sit under the whole confession.", "The hum kept the room inhabited.", "The hum kept pulling me out of the confession."),
    ("dialogue", "She says, \"Keep the dog,\" and he answers, \"The bowl is already by the door.\"", "The exchange told me the marriage was over without a thesis.", "The exchange announced the marriage was over like a caption."),
    ("dialogue", "The lawyer asks three short questions and the witness answers each with the same street name.", "The repetition sounded like fear.", "The repetition sounded like a writing exercise."),
    ("dialogue", "Two soldiers trade the name of a song while filling sandbags, and neither sings it.", "What they withheld was the point.", "What they withheld was a missed beat."),
    ("dialogue", "The child asks why the chairs are stacked, and the uncle says the party already happened.", "That line carried the whole grief for me.", "That line explained the grief and flattened it."),
    ("dialogue", "In the taxi the driver talks about the match, and the passenger only corrects the score.", "The correction sketched a man who cannot let a fact go.", "The correction sketched a man the script wanted me to dislike on schedule."),
    ("dialogue", "The judge reads the date, pauses, and asks the clerk to repeat it.", "The pause made procedure feel human.", "The pause made procedure feel theatrical."),
    ("dialogue", "\"We can still make the ferry,\" she says, already holding both tickets.", "The line had hope I could share.", "The line had hope the images had not earned."),
    ("dialogue", "He describes the painting as \"mostly blue\" and will not name the figure in it.", "The evasion felt like care.", "The evasion felt like coyness."),
    ("dialogue", "The radio host laughs, then reads the cancellation notice without changing tone.", "The professionalism was chilling in a useful way.", "The professionalism was a flat joke."),
    ("dialogue", "At dinner nobody mentions the empty chair, and the youngest finally asks who the glass is for.", "The question opened the scene.", "The question closed the scene with a nudge."),
    ("comparison", "This version of the heist keeps the same blueprint and the same rooftop door as the earlier film.", "The familiarity let the small changes matter.", "The familiarity left the small changes looking timid."),
    ("comparison", "Both brothers wear the father's coat in different scenes, and the sleeves are still too long.", "The shared costume linked them without a speech.", "The shared costume underlined the theme in marker."),
    ("comparison", "The remake uses the original courtroom sketch but moves the verdict to the hallway.", "Moving the verdict made the public moment private, which suited it.", "Moving the verdict threw away the one scene the original understood."),
    ("comparison", "Scene by scene it follows the novel until the pier, where the book has fog and the film has noon.", "The noon light was a real decision.", "The noon light was a random departure."),
    ("comparison", "The first hour is a road movie and the second hour is a hearing, with the same suitcase in both.", "The suitcase made the gear-change feel continuous.", "The suitcase could not hide how abrupt the gear-change was."),
    ("comparison", "She plays the role more quietly than the actress in the trailer, and the lines are unchanged.", "The quieter reading found a person in the lines.", "The quieter reading drained the lines of the only energy they had."),
    ("comparison", "Compared with the director's last film, the town is the same and the festival day is shorter.", "The shorter day gave the rituals weight.", "The shorter day made the rituals look abbreviated."),
    ("comparison", "The documentary and the staged scenes share one interview chair.", "Sharing the chair asked a useful question about what was rehearsed.", "Sharing the chair muddied which claims I was meant to trust."),
    ("comparison", "Where the trailer sells a romance, the cut I saw keeps the same cafe and drops the kiss.", "Dropping the kiss made the attachment clearer.", "Dropping the kiss made the attachment a rumor."),
    ("comparison", "It quotes the western's doorway standoff, then lets a bus block the sightline.", "The blocked sightline was a witty correction.", "The blocked sightline was a joke without a target."),
    ("memory", "I remember the laundromat scene for the way she folds one shirt and leaves the rest in the basket.", "The unfinished basket stayed with me as a choice.", "The unfinished basket stayed with me as an incomplete scene."),
    ("memory", "Years later the image I keep is the coach drawing the play on a fogged window.", "That diagram made the loss specific.", "That diagram made the loss generic."),
    ("memory", "The birthday dinner is ordinary: supermarket cake, one candle, a phone face-down.", "Its ordinariness is what made the later news hurt.", "Its ordinariness is all it had, and the later news felt pasted on."),
    ("memory", "I can still hear the projector in the church basement whenever the narrator mentions August.", "The sound attached the month to a place.", "The sound attached the month to a gimmick."),
    ("memory", "What I retained was not the twist but the clerk restacking the returned tapes.", "The clerk's hands gave the twist a workplace.", "The clerk's hands were a sentimental appendix to the twist."),
    ("memory", "The film leaves me with the smell it keeps describing: diesel and oranges in the truck cab.", "The repetition of that smell worked as memory.", "The repetition of that smell worked as a motif checklist."),
    ("memory", "After it ended I kept the shot of the spare key under the mat, not the confrontation.", "The key was the story's real object.", "The key was a symbol doing the confrontation's job."),
    ("memory", "My notes from the screening mention only the cracked lens in the science-class scene.", "That flaw made the student's view feel physical.", "That flaw drew attention to the prop and away from the student."),
    ("memory", "I forgot the plot points and kept the way the grandmother misnames the station.", "The misnaming was character, not a gag.", "The misnaming was a gag the film repeated until it thinned."),
    ("memory", "The closing song is one the characters never hear, played over the locked shop.", "Hearing it alone made the lock feel final.", "Hearing it alone made the lock feel scored from outside."),
    ("aside", "A side plot follows the night librarian who stamps returns while the main search goes on.", "Her ordinary shift gave the search a city to happen in.", "Her ordinary shift competed with the search and lost."),
    ("aside", "Between chases, the film stops for a man teaching his nephew to tie a tie on a bus.", "The lesson was a breath that sharpened the next chase.", "The lesson was a detour that softened the next chase."),
    ("aside", "The subplot about the borrowed violin never reaches the concert the poster promises.", "Refusing the concert kept the debt personal.", "Refusing the concert kept the debt unresolved in a sloppy way."),
    ("aside", "An aside in the newsroom shows the intern correcting a caption nobody else reads.", "That correction was the film's idea of integrity, and it fit.", "That correction was the film's idea of integrity, and it was tiny."),
    ("aside", "While the trial continues, the camera follows a juror who sketches the stenographer.", "The sketch suggested a mind still present.", "The sketch suggested a mind the script had abandoned."),
    ("aside", "There is a running bit with a misdelivered cactus that appears in three apartments.", "The cactus connected the tenants without a meeting.", "The cactus connected the tenants like a sitcom prop."),
    ("aside", "A brief scene at the border shows the same form being filled by two strangers.", "The duplicate form made the bureaucracy felt.", "The duplicate form made the bureaucracy explained."),
    ("aside", "The film pauses on a chess clock in a park while the argument continues offscreen.", "The clock let me measure the argument.", "The clock upstaged the argument."),
    ("aside", "One insert shows the recipe card burned at the corner, then returns to the unchanged stew.", "The burn mark was enough history.", "The burn mark was a footnote the stew did not need."),
    ("aside", "A neighbor's radio plays the election results through the wall during the breakup.", "The results made the private fight feel dated and real.", "The results made the private fight feel like a backdrop."),
    ("inventory", "The evidence table holds a glove, a ferry ticket, and a child's drawing of a red door.", "Seeing them together made the case tactile.", "Seeing them together made the case a list."),
    ("inventory", "The toolbox contains wire, a saint's medal, and the wrong size of fuse.", "The wrong fuse was a human mistake I recognized.", "The wrong fuse was a plot coupon I saw coming."),
    ("inventory", "On the dashboard: a parking stub, two cigarettes, and a photograph turned face down.", "The face-down photo did the work of a confession.", "The face-down photo postponed a confession the film never delivered."),
    ("inventory", "The classroom drawer has chalk, a confiscated phone, and a list of names with one crossed out.", "The crossed name gave the lesson its stakes.", "The crossed name gave the lesson a mystery-box sticker."),
    ("inventory", "Packing the apartment, they keep the mugs and leave the framed map.", "The choice of what stayed said more than the dialogue.", "The choice of what stayed was the dialogue in object form, and it was obvious."),
    ("inventory", "The surgeon's tray is shown once: gauze, a watch, and a ticket stub from another city.", "The stub suggested a life without a monologue.", "The stub suggested a backstory the film was afraid to write."),
    ("inventory", "Three meals are shown in full: burnt toast, a shared orange, and an untouched banquet plate.", "The sequence charted the relationship in food.", "The sequence charted the relationship like a diagram."),
    ("inventory", "The lost-and-found box contains one red glove and a stack of identical black ones.", "The red glove made the search feel possible.", "The red glove made the search feel rigged."),
    ("inventory", "Credits list the town, the river, and the factory as if they were cast members.", "Treating places as cast suited a film this interested in work.", "Treating places as cast was a closing gimmick."),
    ("inventory", "The notebook pages we can read list only times of day, not feelings.", "The times made the obsession measurable.", "The times made the obsession a prop."),
    ("aftermath", "After the rescue, the street is hosed down and the shop reopens with the same sign.", "Reopening on the same day made survival look like labor.", "Reopening on the same day made survival look like a reset button."),
    ("aftermath", "The epilogue is a single shot of the repaired window with the tape still on it.", "Leaving the tape showed the repair was recent and incomplete.", "Leaving the tape showed the film could not commit to an aftermath."),
    ("aftermath", "Weeks later the team photo hangs in the hall, and one face is turned away.", "The turned face kept the victory from sealing shut.", "The turned face nagged for a sequel beat."),
    ("aftermath", "The letter is delivered in the last scene, and the recipient does not open it on camera.", "Not opening it left the consequence with the character.", "Not opening it left the consequence with the audience as homework."),
    ("aftermath", "After the storm the piano is carried back upstairs by the same four neighbors.", "The carrying was the reconciliation, and it was physical.", "The carrying was the reconciliation, and it was convenient."),
    ("aftermath", "The court steps are empty in the morning, with one leaflet stuck to the rail.", "The leaflet was a trace, not a speech.", "The leaflet was a speech the film pretended not to make."),
    ("aftermath", "A year on, the garden has the same broken chair and a new row of beans.", "The beans argued for time passing without a title card.", "The beans were a title card made of vegetables."),
    ("aftermath", "The band finishes the set, and the drummer stays to coil the cables.", "Staying for the cables gave the triumph a job.", "Staying for the cables tacked a moral onto the triumph."),
    ("aftermath", "When the lights come up, the usher sweeps confetti that the scene itself never showed falling.", "The unswept-then-swept joke acknowledged the show.", "The unswept-then-swept joke apologized for the show."),
    ("aftermath", "The last line is an address read into a phone that has already been hung up.", "Speaking to the dead line felt honest.", "Speaking to the dead line felt arranged."),
]


def main() -> None:
    if len(RAW) != 80:
        raise SystemExit(f"expected 80 pairs, got {len(RAW)}")
    families = {}
    pairs = []
    for i, (family, fact, pos_eval, neg_eval) in enumerate(RAW):
        bridge = BRIDGES[i % len(BRIDGES)]
        families[family] = families.get(family, 0) + 1
        pairs.append({
            "pair_id": f"pair_{i:02d}",
            "family_id": family,
            "bridge": bridge,
            "shared_fact": fact,
            "positive": fact + " " + pos_eval,
            "negative": fact + " " + neg_eval,
        })
    out = Path(__file__).with_name("imdb_matched_pairs_v1.json")
    out.write_text(json.dumps({"n": len(pairs), "bridges": BRIDGES, "family_counts": families, "pairs": pairs}, indent=2))
    print(out, families)


if __name__ == "__main__":
    main()
