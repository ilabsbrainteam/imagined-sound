\version "2.24"

color = #(define-music-function (parser location color) (string?) #{
        \once \override NoteHead.color = #(x11-color color)
        \once \override Stem.color = #(x11-color color)
        \once \override Rest.color = #(x11-color color)
        \once \override Beam.color = #(x11-color color)
     #})

\paper { }

\layout {
  \context {
    \Staff
    \numericTimeSignature
  }
}

\header { }
