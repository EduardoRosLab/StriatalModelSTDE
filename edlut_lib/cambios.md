# CAMBIOS QUE INTRODUCIR

## Variables de estado

```C
static const int max_weight_index = 5;
```

No entiendo la utilidad de la variable de estado `max_weight_index`. ¿Por qué no utilizar una variable local?

## Variables locales de `DopamineSTDPState`

```C
float LTPtau,
float MaxChangeLTP,
float LTDtau,
float MaxChangeLTD,
float base_dopamine
```

- `LTPtau` y `LTDtau` habría que renombrarlas a `tau_p` y `tau_m`, respectivamente. **HECHO**
- `MaxChangeLTP` y `MaxChangeLTD` se sustituyen por `kph`, `kpl`, `kmh`, `kml`.
- `base_dopamine` se sustituye por `da_max` y `da_min`.


## Variables locales de `DopamineSTDPWeightChange`

```C
float MaxChangeLTP;
float tauLTP;
float MaxChangeLTD;
float tauLTD;

float tau_eligibility;
float tau_dopamine;
float increment_dopamine;
float base_dopamine;
```

- `tauLTP` y `tauLTD` se renombran a `tau_p` y `tau_m`
- `MaxChangeLTP` y `MaxChangeLTD` se sustituyen por `kph`, `kpl`, `kmh`, `kml`.
- `base_dopamine` se sustituye por `da_max` y `da_min`


## Parámetros de regla de aprendizaje

```
{'bas_dop': 0.009999999776482582,
 'inc_dop': 0.009999999776482582,
 'max_LTD': 0.032999999821186066,
 'max_LTP': 0.01600000075995922,
 'tau_LTD': 0.10000000149011612,
 'tau_LTP': 0.10000000149011612,
 'tau_dop': 0.30000001192092896,
 'tau_eli': 0.20000000298023224}
```

- `tau_LTP` y `tau_LTD` habría que renombrarlas a `tau_plu` y `tau_min`, respectivamente. **HECHO**
- `max_LTP` y `max_LTD` se sustituyen por `k_plu_hig`, `k_plu_low`, `k_min_hig`, `k_min_low`.
- `bas_da` se sustituye por `dop_max` y `dop_min`.
