---
layout: post
title:  "Belusov Zabontinsky Reactions"
---
Ever since I read the book **Nonlinear Dynamics and Chaos** from Steven Strogatz [^1], 
I grow a fascination for the Belusov-Zabotinsky reaction. This is a chemical reaction
that approaches a equilibrium that is an oscillation, in mathematical terms, a limit cycle.

Of course in practice, the reaction won't osciallte forever, but it will in my simulation below

<h3> Simulation </h3>
<canvas id="sim" width="600" height="450" style="border:1px solid #000000;"></canvas>
<script src="/assets/js/belusov-zabotinksy.js"></script>

The simulation is a simple implementation of the reaction, where the concentration of the reactants is represented by the color of the pixels in the canvas. The reaction is represented by the following set of differential equations:

$$
\begin{align*}
\frac{dA}{dt} &= -\alpha A B + \gamma - A \\
\frac{dB}{dt} &= \alpha A B - \beta B \\
\frac{dC}{dt} &= -\alpha A B + \gamma - C
\end{align*}
$$

Where \\(A\\), \\(B\\) and \\(C\\) are the concentration of the reactants, and \\( \alpha \\), \\( \beta \\) and \\( \gamma \\) are the reaction parameters. 

<!-- <div id="parameters" style="width: 100%;margin: 0 auto;">
  <div id="simulation-parameters" style="width: 80%;display: inline-block;">
    <form id="simulation-form">
      <div id="greeks" style="display: inline-block;">
        <strong>Simulation Parameters</strong>
        <label for="alpha">&#945:</label>
        <input type="text" id="alpha" name="alpha" size="3" value="1.3" style="text-align: center;">
        <label for="beta">&#946:</label>
        <input type="text" id="beta" name="beta" size="3" value="1.0" style="text-align: center;">
        <label for="gamma">&#947:</label>
        <input type="text" id="gamma" name="gamma" size="3" value="1.0" style="text-align: center;">
      </div>
      <br>
      <div id="channels" style="display: inline-block;"">
        <strong>Color Channels </strong>
        <input type="radio" id="one" name="channels" value="1">
        <label for="one">1</label>
        <input type="radio" id="two" name="channels" value="2">
        <label for="two">2</label>
        <input type="radio" id="three" name="channels" value="3" checked>
        <label for="three">3</label>
      </div>
      <br>
      <input type="submit" value="Run" style="height:50px;width:100px;">
    </form>
  </div>
</div> -->

---
[^1]: [Nonlinear Dynamics and Chaos, Steven Strogatz, 1994](https://www.biodyn.ro/course/literatura/Nonlinear_Dynamics_and_Chaos_2018_Steven_H._Strogatz.pdf)