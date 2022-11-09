---
layout: post
title: "Interactive Maps in R"
subtitle: "Creating interactive maps with the Plotly package"
background: '/img/posts/interactive-maps/bg-map.jpg'
---

## Loading in our libraries

``` r
library(plotly)
library(dplyr)
library(readr)
```

## Reading in our data

``` r
states = read_csv("states.csv")

minwage_df = read_csv("Minimum Wage Data.csv") %>%
  inner_join(states, by.x = State, by.x = state) %>%
  select(Year, State, Code, Wage = High.Value) %>%
  mutate(hover = paste0(State, "\n$", Wage))
```

## Setting graph properties

``` r
graph_properties <- list(
  scope = 'usa',
  showland = TRUE,
  landcolor = toRGB("white"),
  color = toRGB("white")
)

font = list(
  family = "DM Sans",
  size = 15,
  color = "black"
)

label = list(
  bgcolor = "#EEEEEE",
  bordercolor = "transparent",
  font = font
)
```

## Making our map

``` r
minwage_graph = plot_geo(minwage_df, 
                         locationmode = "USA-states", 
                         frame = ~Year) %>%
  add_trace(locations = ~Code,
            z = ~Wage,
            zmin = 0,
            zmax = max(minwage_df$Wage),
            color = ~Wage,
            colorscale = "Electric",
            text = ~hover,
            hoverinfo = 'text') %>%
  layout(geo = graph_properties,
         title = "Minimum Wage in the US\n1968 - 2017",
         font = list(family = "DM Sans")) %>%
  config(displayModeBar = FALSE) %>%
  style(hoverlabel = label) %>%
  colorbar(tickprefix = '$')
```


<iframe src="/img/posts/interactive-maps/minwage_graph.html" height="800px" width="100%"></iframe>
