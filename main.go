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

func main() {
	rand.Seed(1)

	set := tc128.NewSet()
	set.Add("a", Size, Size)
	set.Add("b", Size, Size)
	set.Add("c", Size, Size)

	random128 := func(a, b float64) complex128 {
		return complex((b-a)*rand.Float64()+a, (b-a)*rand.Float64()+a)
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

	l1 := tc128.Mul(set.Get("b"), set.Get("c"))
	l2 := tc128.Mul(set.Get("a"), set.Get("c"))
	l3 := tc128.Mul(set.Get("a"), set.Get("b"))
	cost := tc128.Add(
		tc128.Add(
			tc128.Avg(tc128.Quadratic(l1, set.Get("a"))),
			tc128.Avg(tc128.Quadratic(l2, set.Get("b"))),
		),
		tc128.Avg(tc128.Quadratic(l3, set.Get("c"))),
	)

	eta, iterations := complex128(.3+.3i), 1024

	points := make(plotter.XYs, 0, iterations)
	phase := make(plotter.XYs, 0, iterations)
	deta := make(plotter.XYs, 0, iterations)
	detb := make(plotter.XYs, 0, iterations)
	detc := make(plotter.XYs, 0, iterations)
	i := 0
	for i < iterations {
		total := complex128(0)
		set.Zero()

		total += tc128.Gradient(cost).X[0]
		sum := 0.0
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += cmplx.Abs(d) * cmplx.Abs(d)
			}
		}
		norm := float64(math.Sqrt(float64(sum)))
		scaling := float64(1)
		if norm > 1 {
			scaling = 1 / norm
		}

		for _, p := range set.Weights {
			for l, d := range p.D {
				p.X[l] -= eta * d * complex(scaling, 0)
			}
		}

		da := determinant(set.Weights[0].X, Size)
		if cmplx.IsInf(da) {
			break
		}
		a := cmplx.Abs(da)
		if a < 0 {
			a = -a
		}
		if math.IsInf(a, 0) {
			break
		}
		deta = append(deta, plotter.XY{X: float64(i), Y: a})

		db := determinant(set.Weights[1].X, Size)
		if cmplx.IsInf(db) {
			break
		}
		b := cmplx.Abs(db)
		if b < 0 {
			b = -b
		}
		if math.IsInf(b, 0) {
			break
		}
		detb = append(detb, plotter.XY{X: float64(i), Y: b})

		dc := determinant(set.Weights[1].X, Size)
		if cmplx.IsInf(dc) {
			break
		}
		c := cmplx.Abs(dc)
		if c < 0 {
			c = -c
		}
		if math.IsInf(c, 0) {
			break
		}
		detc = append(detc, plotter.XY{X: float64(i), Y: c})

		points = append(points, plotter.XY{X: float64(i), Y: cmplx.Abs(total)})
		phase = append(phase, plotter.XY{X: float64(i), Y: cmplx.Phase(total)})
		fmt.Println(i, cmplx.Abs(total))
		i++
	}

	p := plot.New()

	p.Title.Text = "epochs vs cost"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "cost"

	scatter, err := plotter.NewScatter(points)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "cost.png")
	if err != nil {
		panic(err)
	}

	p = plot.New()

	p.Title.Text = "epochs vs phase"
	p.X.Label.Text = "phase"
	p.Y.Label.Text = "phase"

	scatter, err = plotter.NewScatter(phase)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "phase.png")
	if err != nil {
		panic(err)
	}

	for _, w := range set.Weights {
		fmt.Println(w.X)
	}

	p = plot.New()

	p.Title.Text = "epochs vs det"
	p.X.Label.Text = "epochs"
	p.Y.Label.Text = "det"

	scatter, err = plotter.NewScatter(deta)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{0xFF, 0, 0, 255}
	p.Add(scatter)

	scatter, err = plotter.NewScatter(detb)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{0, 0, 0xFF, 255}
	p.Add(scatter)

	scatter, err = plotter.NewScatter(detc)
	if err != nil {
		panic(err)
	}
	scatter.GlyphStyle.Radius = vg.Length(1)
	scatter.GlyphStyle.Shape = draw.CircleGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{0, 0xFF, 0, 255}
	p.Add(scatter)

	err = p.Save(8*vg.Inch, 8*vg.Inch, "det.png")
	if err != nil {
		panic(err)
	}
}
