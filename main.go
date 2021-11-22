// Copyright 2021 The Trifecta Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"image/color"
	"math"
	"math/cmplx"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/pointlander/gradient/tc128"
)

const (
	// Size is the size of the neuron
	Size = 3
	// Eta is how fast the model will learn
	Eta = .3 + .3i
)

// https://www.geeksforgeeks.org/determinant-of-a-matrix/
func cofactor(mat, temp []complex128, p, q, n int) {
	i, j := 0, 0
	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			if row != p && col != q {
				temp[i*Size+j] = mat[row*Size+col]
				j++
				if j == n-1 {
					j = 0
					i++
				}
			}
		}
	}
}

func determinant(mat []complex128, n int) complex128 {
	if n == 1 {
		return mat[0]
	}
	var d complex128
	temp := make([]complex128, Size*Size)
	sign := complex128(1)
	for f := 0; f < n; f++ {
		cofactor(mat, temp, 0, f, n)
		d += sign * mat[f] * determinant(temp, n-1)
		sign = -sign
	}
	return d
}

// Neuron is a neuron
type Neuron struct {
	Name       int
	Iteration  int
	Set        *tc128.Set
	Rand       *rand.Rand
	E1, E2, E3 tc128.Meta
	Cost       tc128.Meta
	Abs        plotter.XYs
	Phase      plotter.XYs
	Deta       plotter.XYs
	Detb       plotter.XYs
	Detc       plotter.XYs
}

// NewNeuron creates a new neuron
func NewNeuron(name int, seed int64) *Neuron {
	rnd := rand.New(rand.NewSource(seed))

	set := tc128.NewSet()
	set.Add("a", Size, Size)
	set.Add("b", Size, Size)
	set.Add("c", Size, Size)

	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rnd.Float64()+a, (b-a)*rnd.Float64()+a)
	}

	for i := range set.Weights {
		w := set.Weights[i]
		if w.S[1] == 1 {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		} else {
			for i := 0; i < cap(w.X); i++ {
				w.X = append(w.X, random128(-1, 1))
			}
		}
	}

	abs := make(plotter.XYs, 0, 8)
	phase := make(plotter.XYs, 0, 8)
	deta := make(plotter.XYs, 0, 8)
	detb := make(plotter.XYs, 0, 8)
	detc := make(plotter.XYs, 0, 8)

	e1 := tc128.Mul(set.Get("b"), set.Get("c"))
	e2 := tc128.Mul(set.Get("a"), set.Get("c"))
	e3 := tc128.Mul(set.Get("a"), set.Get("b"))
	cost := tc128.Add(
		tc128.Add(
			tc128.Avg(tc128.Quadratic(e1, set.Get("a"))),
			tc128.Avg(tc128.Quadratic(e2, set.Get("b"))),
		),
		tc128.Avg(tc128.Quadratic(e3, set.Get("c"))),
	)

	return &Neuron{
		Name:  name,
		Set:   &set,
		Rand:  rnd,
		E1:    e1,
		E2:    e2,
		E3:    e3,
		Cost:  cost,
		Abs:   abs,
		Phase: phase,
		Deta:  deta,
		Detb:  detb,
		Detc:  detc,
	}
}

// Iterate iterates the model
func (n *Neuron) Iterate() float64 {
	total := complex128(0)
	n.Set.Zero()

	total += tc128.Gradient(n.Cost).X[0]
	sum := 0.0
	for _, p := range n.Set.Weights {
		for _, d := range p.D {
			sum += cmplx.Abs(d) * cmplx.Abs(d)
		}
	}
	norm := float64(math.Sqrt(float64(sum)))
	scaling := float64(1)
	if norm > 1 {
		scaling = 1 / norm
	}

	for _, p := range n.Set.Weights {
		for l, d := range p.D {
			p.X[l] -= Eta * d * complex(scaling, 0)
		}
	}

	da := determinant(n.Set.Weights[0].X, Size)
	if cmplx.IsInf(da) {
		return -1
	}
	a := cmplx.Abs(da)
	if a < 0 {
		a = -a
	}
	if math.IsInf(a, 0) {
		return -1
	}
	n.Deta = append(n.Deta, plotter.XY{X: float64(n.Iteration), Y: a})

	db := determinant(n.Set.Weights[1].X, Size)
	if cmplx.IsInf(db) {
		return -1
	}
	b := cmplx.Abs(db)
	if b < 0 {
		b = -b
	}
	if math.IsInf(b, 0) {
		return -1
	}
	n.Detb = append(n.Detb, plotter.XY{X: float64(n.Iteration), Y: b})

	dc := determinant(n.Set.Weights[1].X, Size)
	if cmplx.IsInf(dc) {
		return -1
	}
	c := cmplx.Abs(dc)
	if c < 0 {
		c = -c
	}
	if math.IsInf(c, 0) {
		return -1
	}
	n.Detc = append(n.Detc, plotter.XY{X: float64(n.Iteration), Y: c})

	n.Abs = append(n.Abs, plotter.XY{X: float64(n.Iteration), Y: cmplx.Abs(total)})
	n.Phase = append(n.Phase, plotter.XY{X: float64(n.Iteration), Y: cmplx.Phase(total)})

	n.Iteration++

	return cmplx.Abs(total)
}

// Graph graphs the properties of the neuron
func (n *Neuron) Graph() {
	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(n.Abs)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_cost.png", n.Name))
	if err != nil {
		panic(err)
	}

	p = plot.New()

	p.Title.Text = "epochs vs phase"
	p.X.Label.Text = "phase"
	p.Y.Label.Text = "phase"

	scatter, err = plotter.NewScatter(n.Phase)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_phase.png", n.Name))
	if err != nil {
		panic(err)
	}

	p = plot.New()

	p.Title.Text = "epochs vs det"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "det"

	scatter, err = plotter.NewScatter(n.Deta)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{0xFF, 0, 0, 255}
	p.Add(scatter)

	scatter, err = plotter.NewScatter(n.Detb)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{0, 0, 0xFF, 255}
	p.Add(scatter)

	scatter, err = plotter.NewScatter(n.Detc)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{0, 0xFF, 0, 255}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("%d_det.png", n.Name))
	if err != nil {
		panic(err)
	}
}

func main() {
	n0, n1 := NewNeuron(0, 1), NewNeuron(1, 2)
	iterations := 1024

	i := 0
	for i < iterations {
		v0 := n0.Iterate()
		v1 := n1.Iterate()
		if v0 > 128 {
			fmt.Println("fire 0")
			w0 := n0.Set.ByName["a"]
			w1 := n1.Set.ByName["a"]
			for j, value := range w0.X {
				w1.X[j] = (w1.X[j] + value) / 2
			}
		}
		if v1 > 128 {
			fmt.Println("fire 1")
			w0 := n0.Set.ByName["a"]
			w1 := n1.Set.ByName["a"]
			for j, value := range w1.X {
				w0.X[j] = (w0.X[j] + value) / 2
			}
		}
		i++
	}

	n0.Graph()
	n1.Graph()
}
