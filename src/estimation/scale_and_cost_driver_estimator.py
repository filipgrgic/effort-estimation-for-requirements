from llm.backend import (
    estimate_prec_prompt,
    estimate_flex_prompt,
    estimate_resl_prompt,
    estimate_rely_prompt,
    estimate_data_prompt,
    estimate_docu_prompt,
    estimate_cplx_prompt,
    estimate_ruse_prompt,
    estimate_time_prompt,
    estimate_stor_prompt,
    estimate_pvol_prompt,
)
import json, math


def estimate_scaling_drivers(reqs: str) -> float:
    sum = 0

    prec_json = json.loads(estimate_prec_prompt(reqs))
    prec_index = calculate_average(prec_json, 3)
    sum += prec_weights[prec_index - 1]

    flex_json = json.loads(estimate_flex_prompt(reqs))
    flex_index = calculate_average(flex_json, 3)
    sum += flex_weights[flex_index - 1]

    resl_json = json.loads(estimate_resl_prompt(reqs))
    resl_index = calculate_average(resl_json, 3)
    sum += resl_weights[resl_index - 1]

    # TEAM and PMAT cannot be estimated from the initial requirements, so their nominal values are used.
    sum += 3.29 + 4.68

    return 0.91 + 0.01 * sum


def estimate_cost_drivers_and_ai_factor(reqs: str) -> tuple[float, int]:
    prod = 1

    # RCPX:
    rcpx_sum = 0

    rely_json = json.loads(estimate_rely_prompt(reqs))
    rcpx_sum += nominal_if_none(rely_json["rely"], 3)

    data_json = json.loads(estimate_data_prompt(reqs))
    rcpx_sum += nominal_if_none(data_json["data"], 3)

    docu_json = json.loads(estimate_docu_prompt(reqs))
    rcpx_sum += nominal_if_none(docu_json["docu"], 3)

    cplx_json = json.loads(estimate_cplx_prompt(reqs))
    cplx = calculate_average(cplx_json, 3)
    rcpx_sum += cplx
    ai_reduction_factor = determine_ai_reduction_factor(cplx)

    rcpx_index = determine_rcpx_weight(rcpx_sum)
    prod *= rcpx_weights[rcpx_index]

    # RUSE:
    ruse_json = json.loads(estimate_ruse_prompt(reqs))
    ruse_index = nominal_if_none(ruse_json["ruse"], 3)
    if ruse_index < 2 or ruse_index > 6:
        raise ValueError("RUSE rating must be between 2 and 6")
    prod *= ruse_weights[ruse_index - 2]

    # PDIF:
    pdif_sum = 0

    time_json = json.loads(estimate_time_prompt(reqs))
    pdif_sum += nominal_if_none(time_json["time"], 3)

    stor_json = json.loads(estimate_stor_prompt(reqs))
    pdif_sum += nominal_if_none(stor_json["stor"], 3)

    pvol_json = json.loads(estimate_pvol_prompt(reqs))
    pdif_sum += nominal_if_none(pvol_json["pvol"], 3)

    pdif_index = determine_pdif_weight(pdif_sum)
    prod *= pdif_weights[pdif_index]

    # PERS, PREX, FCIL and SCED cannot be estimated from the initial requirements, so their nominal
    # multiplier 1.0 is used.

    return prod, ai_reduction_factor


def calculate_average(json: dict, nominal: int) -> int:
    sum = 0
    for v in json.values():
        if v is not None:
            sum += v
        else:
            sum += nominal

    return round_to_nominal(sum / len(json), nominal)


def round_to_nominal(x: float, nominal: int) -> int:
    if not x.is_integer():
        if x < nominal:
            x = math.ceil(x)
        else:
            x = math.floor(x)

    return int(x)


def nominal_if_none(value: int | None, nominal: int) -> int:
    return nominal if value is None else value


def determine_rcpx_weight(sum: int) -> int:
    if sum < 5:
        raise ValueError("The sum of the RCPX Ratings must not be smaller than 5.")
    elif sum >= 5 and sum <= 6:
        return 0
    elif sum >= 7 and sum <= 8:
        return 1
    elif sum >= 9 and sum <= 11:
        return 2
    elif sum == 12:
        return 3
    elif sum >= 13 and sum <= 15:
        return 4
    elif sum >= 16 and sum <= 18:
        return 5
    elif sum >= 19 and sum <= 21:
        return 6
    else:
        raise ValueError("The sum of the RCPX Ratings must not be higher than 21.")


def determine_pdif_weight(sum: int) -> int:
    if sum < 8:
        raise ValueError("The sum of the PDIF Ratings must not be smaller than 8.")
    elif sum == 8:
        return 0
    elif sum == 9:
        return 1
    elif sum >= 10 and sum <= 12:
        return 2
    elif sum >= 13 and sum <= 15:
        return 3
    elif sum >= 16 and sum <= 17:
        return 4
    else:
        raise ValueError("The sum of the PDIF Ratings must not be higher than 17.")


def determine_ai_reduction_factor(cplx: int) -> int:
    if cplx <= 2:
        return 5
    elif cplx > 4:
        return 20

    return 10


# scale factor weights:
prec_weights = [6.2, 4.96, 3.72, 2.48, 1.24, 0.0]
flex_weights = [5.07, 4.05, 3.04, 2.03, 1.01, 0.0]
resl_weights = [7.07, 5.65, 4.24, 2.83, 1.41, 0.0]

# cost driver weights:
rcpx_weights = [0.49, 0.6, 0.83, 1.0, 1.33, 1.91, 2.72]
ruse_weights = [0.95, 1.0, 1.07, 1.15, 1.24]
pdif_weights = [0.87, 1.0, 1.29, 1.81, 2.61]
