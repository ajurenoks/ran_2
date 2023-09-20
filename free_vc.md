

```mermaid

flowchart TD
style x1 fill:#fff, stroke-width:0px
style a fill:#fff, stroke-width:0px
style b fill:#fff, stroke-width:0px
style y fill:#fff, stroke-width:0px
style xmel fill:#fff, stroke-width:0px
style z1 fill:#fff, stroke-width:0px
style RF fill:#fff, stroke-width:0px
style g fill:#fff, stroke-width:0px
style b1 fill:#fff, stroke-width:0px
style a1 fill:#fff, stroke-width:0px
style z fill:#fff, stroke-width:0px
style yp fill:#fff, stroke-width:0px
style xin fill:#fff, stroke-width:0px
subgraph zid[Prior encoder]
    WavLM --> x1[X<sub>ssl</sub>]
    x1[X<sub>ssl</sub>] --> BE[Bottleneck extractor]
    BE[Bottleneck extractor] --> a[μ<sub>θ</sub>]
    BE[Bottleneck extractor] --> b[σ<sub>θ</sub>]
    F[Flow]--> z[z']
    end
    z1[z]--> dec[Decoder]
    xmel[x<sub>mel</sub>]-->SBDA[SR-based data agumentation]
    SBDA[SR-based data agumentation] --> y[y']
    y[y'] --> WavLM
    
    xmel[x<sub>mel</sub>] --> SE[Speaker encoder]
    SE[Speaker encoder] --> g[g]
    g[g] --> F[Flow]
    z1[z] --> F[Flow]
    g[g] --> PE[Posterior encoder]
    xin[x<sub>iin</sub>] --> PE[Posterior encoder]
    PE[Posterior encoder] --> a1[μ<sub>Φ</sub>]
    PE[Posterior encoder] --> z1[z]
    PE[Posterior encoder] --> b1[σ<sub>Φ</sub>]
    g[g] --> dec[Decoder]
    
    dec[Decoder] --> yp[ŷ]
    yp[ŷ] --> DS[Discriminator]
    DS[Discriminator] --> RF[Real/ Fake]
    
    
 ``` 