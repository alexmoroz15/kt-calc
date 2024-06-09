"""Calculator for calculating the number of hits or amount of damage from an attack in Kill Team"""

from dataclasses import dataclass
import math
from enum import Enum


@dataclass
class Outcome:
    """Describes the outcome of either an attack or defense roll."""

    num_hits: int
    num_crits: int

    def __post_init__(self):
        assert self.num_hits >= 0
        assert self.num_crits >= 0

    def __eq__(self, value: object) -> bool:
        if isinstance(value, Outcome):
            return self.num_hits == value.num_hits and self.num_crits == value.num_crits
        return False

    def __hash__(self) -> int:
        return hash(self.num_hits) ^ hash(self.num_crits)


@dataclass
class NetOutcome:
    """Outcome after defense dice have been applied. Useful for applying crit effects."""

    net_outcome: Outcome
    original_outcome: Outcome

    def __eq__(self, value: object) -> bool:
        if isinstance(value, NetOutcome):
            return (
                self.net_outcome == value.net_outcome
                and self.original_outcome == value.original_outcome
            )
        return False

    def __hash__(self) -> int:
        return hash(self.net_outcome) ^ hash(self.original_outcome)


class RerollStrategy(Enum):
    """Describes which dice the player will re-roll when given the option of re-rolling"""

    NO_REROLL = 0  # No idea why you'd do this, but included for completeness
    REROLL_MISSES = 1
    FISH_FOR_CRITS = 2
    TRY_TO_MISS = 3


@dataclass
class DiceParams:
    """Describes number and dice behavior for either attacker or defender"""

    num_dice: int
    miss_chance: float
    hit_chance: float
    crit_chance: float
    reroll_given_miss_chance: float
    reroll_given_hit_chance: float
    reroll_given_crit_chance: float

    def __post_init__(self):
        assert self.num_dice >= 0
        assert all(
            0 <= x <= 1
            for x in [
                self.miss_chance,
                self.hit_chance,
                self.crit_chance,
                self.reroll_given_miss_chance,
                self.reroll_given_hit_chance,
                self.reroll_given_crit_chance,
            ]
        )
        assert math.isclose(self.miss_chance + self.hit_chance + self.crit_chance, 1.0)


@dataclass
class DamageParams:
    """Describes how much damage the attacker's weapon deals. Used for allocating saves."""

    normal_damage: int
    crit_damage: int


@dataclass
class KillTeamCalculatorParams:
    attack_dice_params: DiceParams
    defense_dice_params: DiceParams
    damage_params: DamageParams


def calculate_dice_chances(dice_params: DiceParams) -> dict[Outcome, float]:
    # print(f"dice_params: {dice_params}")
    hit_chances: dict[Outcome, float] = {}
    n = dice_params.num_dice
    for num_hits in range(n + 1):
        for num_crits in range(n - num_hits + 1):
            assert num_hits + num_crits <= n
            num_misses = n - num_hits - num_crits
            outcome = Outcome(num_hits, num_crits)
            hit_chances[outcome] = math.prod(
                [
                    (dice_params.num_crit_states / 6) ** num_crits,
                    (dice_params.num_hit_states / 6) ** num_hits,
                    (dice_params.num_miss_states / 6) ** num_misses,
                    math.comb(n, num_hits),
                    math.comb(n - num_hits, num_crits),
                ]
            )
    assert abs(sum(hit_chances.values()) - 1) < 0.0001
    return hit_chances


class KillTeamCalculator:
    def __init__(self, params):
        self.params = params

    params: KillTeamCalculatorParams

    def apply_defense_dice_to_hit_outcome(
        self, hit_outcome: Outcome, defense_outcome: Outcome
    ) -> Outcome:
        # I don't have a CS degree, so just try every combination to see which outcome
        # results in the least damage.

        best_outcome: Outcome | None = None
        lowest_damage = math.inf

        for num_normal_to_crit in range(defense_outcome.num_hits // 2 + 1):
            normal_damage = self.params.damage_params.normal_damage
            crit_damage = self.params.damage_params.crit_damage
            cur_defense_outcome = Outcome(
                num_hits=defense_outcome.num_crits + num_normal_to_crit,
                num_crits=defense_outcome.num_hits - num_normal_to_crit * 2,
            )

            num_crits = max(hit_outcome.num_crits - cur_defense_outcome.num_crits, 0)
            crit_carryover = max(-num_crits, 0)
            num_hits = max(
                hit_outcome.num_hits - cur_defense_outcome.num_hits - crit_carryover, 0
            )

            expected_damage = normal_damage * num_hits + crit_damage * num_crits
            current_outcome = Outcome(num_hits, num_crits)
            if expected_damage < lowest_damage:
                lowest_damage = expected_damage
                best_outcome = current_outcome

        assert best_outcome is not None
        return best_outcome

    def calculate_hit_chances(self) -> dict[Outcome, float]:
        return calculate_dice_chances(self.params.attack_dice_params)

    def calculate_defense_chances(self) -> dict[Outcome, float]:
        return calculate_dice_chances(self.params.defense_dice_params)

    def calculate_net_hit_chances(self) -> dict[Outcome, float]:
        hit_chances = self.calculate_hit_chances()
        defence_chances = self.calculate_defense_chances()

        net_hit_chances: dict[Outcome, float] = {}
        for hit_outcome, hit_chance in hit_chances.items():
            for defense_outcome, defense_chance in defence_chances.items():
                net_hit_outcome = self.apply_defense_dice_to_hit_outcome(
                    hit_outcome, defense_outcome
                )
                net_hit_chances[net_hit_outcome] = (
                    net_hit_chances.get(net_hit_outcome, 0)
                    + hit_chance * defense_chance
                )
        return net_hit_chances

    def calculate_damage_chances(self) -> dict[int, float]:
        hit_chances = self.calculate_net_hit_chances()
        damage_chances: dict[int, float] = {}
        for outcome, chance in hit_chances.items():
            damage = (
                outcome.num_hits * self.params.damage_params.normal_damage
                + outcome.num_crits * self.params.damage_params.crit_damage
            )
            damage_chances[damage] = damage_chances.get(damage, 0) + chance
        return damage_chances


class KillTeamCalculatorBuilder:
    bs: int | None = None
    num_attacks: int | None = None
    normal_damage: int | None = None
    crit_damage: int | None = None
    sv: int | None = None
    lethal: int | None = None
    ap: int | None = None

    def with_ballistic_skill(self, bs: int):
        assert 1 < bs < 7
        self.bs = bs
        return self

    def with_num_attacks(self, num_attacks: int):
        assert num_attacks >= 1
        self.num_attacks = num_attacks
        return self

    def with_normal_damage(self, normal_damage: int):
        assert normal_damage >= 0
        self.normal_damage = normal_damage
        return self

    def with_crit_damage(self, crit_damage: int):
        assert crit_damage >= 0
        self.crit_damage = crit_damage
        return self

    def with_save_characteristic(self, sv: int):
        assert 1 < sv < 7
        self.sv = sv
        return self

    def with_lethal_x(self, x: int):
        assert 1 < x < 7
        self.lethal = x
        return self

    def with_mw_x(self, x: int):
        raise NotImplementedError("MW x is not supported yet")

    def with_free_normal_saves(self, num_free_saves: int):
        raise NotImplementedError("Having free saves is not supported yet")

    def with_ap_x(self, x: int):
        assert 0 <= x <= 2
        self.ap = x
        return self

    def with_p_x(self, x: int):
        raise NotImplementedError("P x is not supported yet")

    def with_ceaseless(self, ceaseless: bool):
        raise NotImplementedError("Ceaseless is not supported yet")

    def with_relentless(self, relentless: bool):
        raise NotImplementedError("Relentless is not supported yet")

    def with_balances(self, balanced: bool):
        raise NotImplementedError("Balanced is not supported yet")

    def build(self) -> KillTeamCalculator:
        assert self.bs is not None
        assert self.num_attacks is not None
        assert self.normal_damage is not None
        assert self.crit_damage is not None
        assert self.sv is not None
        params = KillTeamCalculatorParams(
            attack_dice_params=DiceParams(
                num_dice=self.num_attacks,
                num_hit_states=(
                    6 - self.bs if self.lethal is None else self.lethal - self.bs
                ),
                num_crit_states=1 if self.lethal is None else 7 - self.lethal,
                num_miss_states=self.bs - 1,
            ),
            defense_dice_params=DiceParams(
                num_dice=3 if self.ap is None else 3 - self.ap,
                num_hit_states=6 - self.sv,
                num_crit_states=1,
                num_miss_states=self.sv - 1,
            ),
            damage_params=DamageParams(
                normal_damage=self.normal_damage,
                crit_damage=self.crit_damage,
            ),
        )
        return KillTeamCalculator(params)


def main():
    # for save_characteristic in range(2, 7):
    for save_characteristic in [5]:
        calculator_fleshborer = (
            KillTeamCalculatorBuilder()
            .with_ballistic_skill(4)
            .with_num_attacks(4)
            .with_normal_damage(3)
            .with_crit_damage(4)
            .with_save_characteristic(save_characteristic)
            .build()
        )

        calculator_spinefists = (
            KillTeamCalculatorBuilder()
            .with_ballistic_skill(3)
            .with_num_attacks(4)
            .with_normal_damage(2)
            .with_crit_damage(3)
            .with_save_characteristic(save_characteristic)
            .build()
        )

        calculator_deathspitter = (
            KillTeamCalculatorBuilder()
            .with_ballistic_skill(4)
            .with_num_attacks(5)
            .with_normal_damage(4)
            .with_crit_damage(5)
            .with_save_characteristic(save_characteristic)
            .build()
        )

        calculator_bolter = (
            KillTeamCalculatorBuilder()
            .with_ballistic_skill(3)
            .with_num_attacks(4)
            .with_normal_damage(3)
            .with_crit_damage(4)
            .with_save_characteristic(save_characteristic)
            .build()
        )

        for calculator, weapon_name in [
            (calculator_fleshborer, "fleshborer"),
            (calculator_spinefists, "spinefists"),
            (calculator_deathspitter, "deathspitter"),
            (calculator_bolter, "bolter"),
        ]:
            print(f"evaluating: {weapon_name} against SV {save_characteristic}+")
            damage_chances: dict[int, float] = calculator.calculate_damage_chances()

            def sum_above_threshold(
                chance_map: dict[int, float], threshold: int
            ) -> float:
                return sum(
                    chance for (val, chance) in chance_map.items() if val >= threshold
                )

            for x in range(1, 8):
                print(
                    f"Chance to deal {x}+ damage: {sum_above_threshold(damage_chances, x)}"
                )

            prob_to_deal_damage = [
                (0, damage_chances.get(0, 0)),
                (1, damage_chances.get(1, 0)),
                (2, damage_chances.get(2, 0)),
                (3, damage_chances.get(3, 0)),
                (4, damage_chances.get(4, 0)),
                (5, damage_chances.get(5, 0)),
                (6, damage_chances.get(6, 0)),
                (7, sum_above_threshold(damage_chances, 7)),
            ]

            chance_to_deal_7 = sum(
                x * y
                for (x, y) in [
                    (c1, c2)
                    for (d1, c1) in prob_to_deal_damage
                    for (d2, c2) in prob_to_deal_damage
                    if d1 + d2 >= 7
                ]
            )
            print(f"Chance to kill a guardsman with two shots: {chance_to_deal_7}")


if __name__ == "__main__":
    main()
