#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
해양 강철 재킷 플랫폼의 희생 양극 음극 보호(BEM 기반) 예제 시뮬레이터.
논문에서 제시된 지배 방정식(∇²ϕ=0)과 경계 적분 형식을 단순화한 코드로,
4개의 다리와 3개 층을 갖는 구조를 경계 요소로 이산화하여 전위/전류 밀도를 해석한다.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt  # type: ignore  # pylint: disable=import-error
    import matplotlib.font_manager as fm  # type: ignore
except ImportError as exc:  # pragma: no cover - 의존성 누락 시 명확한 오류 제공
    raise ImportError(
        "matplotlib이 설치되어 있어야 시각화가 가능합니다. "
        "pip install matplotlib 명령으로 설치한 뒤 다시 실행하세요."
    ) from exc

import numpy as np

# 한글 폰트 설정 (OS별 자동 감지)
import platform
system = platform.system()
if system == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
elif system == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:  # Linux
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


# --------------------------------------------------------------------------------------
# 데이터 구조 정의
# --------------------------------------------------------------------------------------

@dataclass
class BoundaryElement:
    """경계 요소 하나를 표현하는 자료 구조."""

    center: np.ndarray
    normal: np.ndarray
    area: float
    kind: str  # 'steel', 'anode', 'insulator'
    tag: str = ""
    fixed_potential: Optional[float] = None
    leg_index: Optional[int] = None
    height: float = 0.0


@dataclass
class SimulationResult:
    """솔버 결과를 저장하기 위한 자료 구조."""

    potentials: np.ndarray
    fluxes: np.ndarray
    iterations: int
    converged: bool
    residual_history: List[float]


@dataclass
class SimulationConfig:
    """시뮬레이션과 양극 소모 예측에 필요한 파라미터 모음."""

    base_span: float = 0.8
    top_span: float = 0.3
    height: float = 1.5
    num_layers: int = 3
    leg_radius: float = 0.045
    coating_fraction: float = 0.2
    tafel_a: float = 0.12
    tafel_b: float = -0.78
    anode_area: float = 0.025
    anode_potential: float = -1.05
    anode_thickness: float = 0.08  # m
    anode_density: float = 2700.0  # kg/m^3 (Al 합금)
    anode_capacity: float = 2500.0  # Ah/kg
    anode_utilization: float = 0.9  # 사용 가능한 질량 비율
    service_days: float = 365.0
    solver_tol: float = 1e-7
    solver_max_iter: int = 50
    phi_init: float = -0.9
    q_init: float = 80.0
    anode_height_fractions: Tuple[float, ...] = (0.35, 0.55)
    anode_legs: Tuple[int, ...] = (0, 1, 2, 3)
    dynamic_mode: bool = False
    dynamic_time_step_days: float = 30.0
    minimum_anode_thickness: float = 0.002
    stop_when_anodes_depleted: bool = True


@dataclass
class AnodeLifeState:
    """동적 해석 시 각 양극의 남은 질량/두께 상태를 추적한다."""

    element_index: int
    tag: str
    initial_area: float
    initial_thickness: float
    density: float
    usable_mass: float
    remaining_mass: float
    current_area: float
    is_active: bool = True

    @property
    def thickness(self) -> float:
        if self.current_area <= 0 or self.density <= 0:
            return 0.0
        return self.remaining_mass / (self.density * self.current_area)

    @property
    def consumed_fraction(self) -> float:
        if self.usable_mass <= 0:
            return 1.0 if not self.is_active else 0.0
        return 1.0 - (self.remaining_mass / self.usable_mass)

    @property
    def volume_fraction(self) -> float:
        if self.usable_mass <= 0:
            return 0.0
        return max(self.remaining_mass / self.usable_mass, 0.0)

    def shrink_geometry(self) -> float:
        """
        질량 비율^(1/3)을 선형 축소 인자로 사용해
        두께·면적이 동시에 줄어드는 단순 모델을 적용한다.
        """

        if self.usable_mass <= 0 or self.initial_area <= 0 or self.initial_thickness <= 0:
            return 0.0

        linear_scale = self.volume_fraction ** (1.0 / 3.0)
        self.current_area = self.initial_area * (linear_scale**2)
        return linear_scale


@dataclass
class DynamicStepRecord:
    """동적 해석 각 스텝의 결과와 양극 상태를 저장."""

    day: float
    step_days: float
    result: SimulationResult
    snapshot: List[Dict[str, float]]


# --------------------------------------------------------------------------------------
# 유틸리티 함수
# --------------------------------------------------------------------------------------

def normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """벡터를 정규화한다."""
    norm = np.linalg.norm(vec)
    if norm < eps:
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def square_corners(span: float) -> np.ndarray:
    """
    정사각형 코너 좌표를 반환한다.
    span: 변 길이 (m)
    """
    half = span / 2.0
    return np.array(
        [
            [half, half],
            [-half, half],
            [-half, -half],
            [half, -half],
        ]
    )


# --------------------------------------------------------------------------------------
# 메쉬 생성
# --------------------------------------------------------------------------------------

class JacketMeshBuilder:
    """
    4-다리, 3-층 재킷 플랫폼의 경계 요소 메쉬를 생성한다.
    - 각 다리는 원통을 둘러싼 등가 단일 패널(상수 요소)로 단순화
    - 상부로 갈수록 스팬이 줄어드는 형상을 가정
    - 상부 일정 구간은 절연 코팅으로 처리 가능
    """

    def __init__(
        self,
        base_span: float = 0.8,
        top_span: float = 0.3,
        height: float = 1.2,
        num_layers: int = 3,
        leg_radius: float = 0.04,
        coating_fraction: float = 0.15,
    ) -> None:
        self.base_span = base_span
        self.top_span = top_span
        self.height = height
        self.num_layers = num_layers
        self.leg_radius = leg_radius
        self.coating_fraction = coating_fraction

    def build(self) -> List[BoundaryElement]:
        if self.num_layers < 1:
            raise ValueError("num_layers는 1 이상이어야 합니다.")

        base_xy = square_corners(self.base_span)
        top_xy = square_corners(self.top_span)
        z_nodes = np.linspace(0.0, self.height, self.num_layers + 1)

        elements: List[BoundaryElement] = []
        for leg_idx in range(4):
            for layer in range(self.num_layers):
                z0, z1 = z_nodes[layer], z_nodes[layer + 1]
                zc = 0.5 * (z0 + z1)
                height_ratio = zc / self.height
                xy_bottom = base_xy[leg_idx]
                xy_top = top_xy[leg_idx]
                xy = (1 - height_ratio) * xy_bottom + height_ratio * xy_top

                center = np.array([xy[0], xy[1], zc])
                normal = normalize(np.array([xy[0], xy[1], 0.0]))
                segment_height = z1 - z0
                area = 2.0 * math.pi * self.leg_radius * segment_height

                kind = "steel"
                if (
                    self.coating_fraction > 0.0
                    and zc >= (1.0 - self.coating_fraction) * self.height
                ):
                    kind = "insulator"

                elements.append(
                    BoundaryElement(
                        center=center,
                        normal=normal,
                        area=area,
                        kind=kind,
                        tag=f"leg{leg_idx}_layer{layer}",
                        leg_index=leg_idx,
                        height=zc,
                    )
                )
        return elements


def place_sacrificial_anodes(
    elements: List[BoundaryElement],
    total_height: float,
    specs: Sequence[Dict[str, float]],
    protrusion: float = 0.06,
) -> None:
    """
    희생 양극 패널을 기존 요소에 부착한다.
    specs 항목 예시: {"leg":0, "height_fraction":0.4, "area":0.02, "potential":-1.05}
    height_fraction 대신 abs_height를 줄 수도 있다.
    """

    for spec in specs:
        leg = int(spec.get("leg", 0))
        if not 0 <= leg < 4:
            raise ValueError("leg index는 0~3 범위여야 합니다.")

        if "abs_height" in spec:
            target_height = float(spec["abs_height"])
        elif "height_fraction" in spec:
            target_height = float(spec["height_fraction"]) * total_height
        else:
            raise ValueError("height_fraction 또는 abs_height를 지정해야 합니다.")

        area = float(spec.get("area", 0.02))
        potential = float(spec.get("potential", -1.05))

        candidates = [
            el for el in elements if el.leg_index == leg and el.kind != "anode"
        ]
        if not candidates:
            raise RuntimeError("지정한 다리에 배치할 요소가 없습니다.")

        target_element = min(
            candidates, key=lambda el: abs(el.height - target_height)
        )
        offset = float(spec.get("offset", protrusion))
        new_center = target_element.center + target_element.normal * offset

        elements.append(
            BoundaryElement(
                center=new_center,
                normal=target_element.normal.copy(),
                area=area,
                kind="anode",
                tag=f"anode_leg{leg}_{target_height:.2f}",
                fixed_potential=potential,
                leg_index=leg,
                height=target_height,
            )
        )


def generate_anode_specs(config: SimulationConfig) -> List[Dict[str, float]]:
    """구성 파라미터를 바탕으로 다리별 양극 배치 정보를 생성한다."""

    specs: List[Dict[str, float]] = []
    for leg in config.anode_legs:
        for frac in config.anode_height_fractions:
            specs.append(
                {
                    "leg": leg,
                    "height_fraction": frac,
                    "area": config.anode_area,
                    "potential": config.anode_potential,
                }
            )
    return specs


def build_platform_elements(config: SimulationConfig) -> Tuple[List[BoundaryElement], JacketMeshBuilder]:
    """플랫폼 메쉬와 양극을 생성해 반환한다."""

    builder = JacketMeshBuilder(
        base_span=config.base_span,
        top_span=config.top_span,
        height=config.height,
        num_layers=config.num_layers,
        leg_radius=config.leg_radius,
        coating_fraction=config.coating_fraction,
    )
    elements = builder.build()
    anode_specs = generate_anode_specs(config)
    place_sacrificial_anodes(elements, builder.height, anode_specs)
    return elements, builder


# --------------------------------------------------------------------------------------
# BEM 행렬 생성
# --------------------------------------------------------------------------------------

def assemble_bem_matrices(elements: Sequence[BoundaryElement]) -> Tuple[np.ndarray, np.ndarray]:
    """
    상수 요소(constant element) 가정을 이용해
    H·ϕ = G·q 시스템 행렬을 구성한다.
    """
    num = len(elements)
    if num == 0:
        raise ValueError("요소가 없습니다.")

    H = np.zeros((num, num), dtype=float)
    G = np.zeros((num, num), dtype=float)

    for i, obs in enumerate(elements):
        ri = obs.center
        ni = obs.normal
        for j, src in enumerate(elements):
            if i == j:
                # 매끄러운 경계에 대해 자체 항은 0.5 로 수렴
                H[i, j] = 0.5
                char_len = max(math.sqrt(src.area / math.pi), 1e-6)
                G[i, j] = src.area / (2.0 * math.pi * char_len)
                continue

            r_vec = ri - src.center
            dist = np.linalg.norm(r_vec)
            if dist < 1e-9:
                dist = 1e-9

            G[i, j] = src.area / (4.0 * math.pi * dist)
            normal_dot = np.dot(ni, r_vec)
            H[i, j] = -src.area * normal_dot / (4.0 * math.pi * dist**3)
    return H, G


# --------------------------------------------------------------------------------------
# 분극 모델 (비선형 경계 조건)
# --------------------------------------------------------------------------------------

class PolarizationModel:
    """
    강재 표면의 타펠 방정식 기반 분극 특성.
    ϕ = -a·log10(|i|) + b  =>  |i| = 10 ** ((b - ϕ) / a)
    """

    def __init__(
        self,
        tafel_a: float = 0.12,
        tafel_b: float = -0.80,
        min_current: float = 1e-6,
        max_current: float = 300.0,
    ) -> None:
        self.tafel_a = tafel_a
        self.tafel_b = tafel_b
        self.min_current = min_current
        self.max_current = max_current

    def _tunnel_current(self, phi: float) -> float:
        exponent = (self.tafel_b - phi) / self.tafel_a
        current = 10.0**exponent
        return current

    def current_density(self, phi: float) -> float:
        """강재 경계의 전류 밀도 (음극 전류 → 음수)."""
        current = self._tunnel_current(phi)
        limited = max(self.min_current, min(current, self.max_current))
        return -limited

    def derivative(self, phi: float) -> float:
        """
        dq/dϕ. 뉴턴-랩슨 자코비안 계산에 필요.
        클리핑 구간에서는 민감도가 감소하도록 0으로 처리.
        """
        current = self._tunnel_current(phi)
        if current <= self.min_current or current >= self.max_current:
            return 0.0
        return (math.log(10.0) / self.tafel_a) * current


# --------------------------------------------------------------------------------------
# 비선형 BEM 솔버 (뉴턴 반복)
# --------------------------------------------------------------------------------------

class CathodicProtectionSolver:
    """Hϕ=Gq 시스템과 비선형 경계 조건을 결합한 솔버."""

    def __init__(
        self,
        elements: List[BoundaryElement],
        polarization: PolarizationModel,
    ) -> None:
        self.elements = elements
        self.polarization = polarization
        self.H, self.G = assemble_bem_matrices(elements)

        self.phi_unknown_idx = [
            idx
            for idx, el in enumerate(elements)
            if not (el.kind == "anode" and el.fixed_potential is not None)
        ]
        self.q_unknown_idx = [idx for idx, el in enumerate(elements) if el.kind == "anode"]

        if len(self.phi_unknown_idx) + len(self.q_unknown_idx) != len(elements):
            raise RuntimeError(
                "경계 조건 설정 오류: 미지수 수가 요소 수와 일치하지 않습니다."
            )

        self.phi_index_map = {gidx: lidx for lidx, gidx in enumerate(self.phi_unknown_idx)}
        self.q_index_map = {
            gidx: lidx for lidx, gidx in enumerate(self.q_unknown_idx)
        }

    def _split_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phi_segment = state[: len(self.phi_unknown_idx)]
        q_segment = state[len(self.phi_unknown_idx) :]
        return phi_segment, q_segment

    def _assemble_fields(
        self, state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        phi_segment, q_segment = self._split_state(state)
        num = len(self.elements)
        phi_all = np.zeros(num, dtype=float)
        q_all = np.zeros(num, dtype=float)

        for idx, local in self.phi_index_map.items():
            phi_all[idx] = phi_segment[local]

        for idx, element in enumerate(self.elements):
            if element.kind == "anode" and element.fixed_potential is not None:
                phi_all[idx] = element.fixed_potential

        for idx, element in enumerate(self.elements):
            if element.kind == "steel":
                q_all[idx] = self.polarization.current_density(phi_all[idx])
            elif element.kind == "insulator":
                q_all[idx] = 0.0
            elif element.kind == "anode":
                local = self.q_index_map[idx]
                q_all[idx] = q_segment[local]
            else:
                raise ValueError(f"알 수 없는 요소 종류: {element.kind}")
        return phi_all, q_all

    def _jacobian(self, phi_all: np.ndarray) -> np.ndarray:
        num = len(self.elements)
        total_unknown = len(self.phi_unknown_idx) + len(self.q_unknown_idx)
        J = np.zeros((num, total_unknown), dtype=float)

        for col, gidx in enumerate(self.phi_unknown_idx):
            column = self.H[:, gidx].copy()
            element = self.elements[gidx]
            if element.kind == "steel":
                dq_dphi = self.polarization.derivative(phi_all[gidx])
                column -= self.G[:, gidx] * dq_dphi
            J[:, col] = column

        offset = len(self.phi_unknown_idx)
        for col, gidx in enumerate(self.q_unknown_idx, start=offset):
            J[:, col] = -self.G[:, gidx]
        return J

    def solve(
        self,
        max_iter: int = 40,
        tol: float = 1e-6,
        phi_init: float = -0.90,
        q_init: float = 50.0,
    ) -> SimulationResult:
        phi_guess = np.full(len(self.phi_unknown_idx), phi_init, dtype=float)
        q_guess = np.full(len(self.q_unknown_idx), q_init, dtype=float)
        state = np.concatenate([phi_guess, q_guess])

        residual_history: List[float] = []
        converged = False

        for iteration in range(1, max_iter + 1):
            phi_all, q_all = self._assemble_fields(state)
            residual = self.H @ phi_all - self.G @ q_all
            res_norm = np.linalg.norm(residual, ord=np.inf)
            residual_history.append(res_norm)

            if res_norm < tol:
                converged = True
                break

            J = self._jacobian(phi_all)
            try:
                delta = np.linalg.solve(J, -residual)
            except np.linalg.LinAlgError:
                delta, *_ = np.linalg.lstsq(J, -residual, rcond=None)

            # 감쇠 인자(라인 서치)로 발산 방지
            damping = 1.0
            for _ in range(5):
                candidate_state = state + damping * delta
                _, candidate_q = self._assemble_fields(candidate_state)
                if np.all(np.isfinite(candidate_q)):
                    state = candidate_state
                    break
                damping *= 0.5
            else:
                raise RuntimeError("뉴턴 업데이트가 실패했습니다.")

        phi_all, q_all = self._assemble_fields(state)
        return SimulationResult(
            potentials=phi_all,
            fluxes=q_all,
            iterations=iteration,
            converged=converged,
            residual_history=residual_history,
        )


# --------------------------------------------------------------------------------------
# 후처리 및 시각화
# --------------------------------------------------------------------------------------

def summarize_result(elements: Sequence[BoundaryElement], result: SimulationResult) -> None:
    phi = result.potentials
    q = result.fluxes

    steel_idx = [i for i, el in enumerate(elements) if el.kind == "steel"]
    anode_idx = [i for i, el in enumerate(elements) if el.kind == "anode"]

    print("=== 계산 요약 ===")
    print(f"수렴 여부: {result.converged} (반복 {result.iterations}회, 최종 residual={result.residual_history[-1]:.3e})")
    print(f"전위 범위: {phi.min():.3f} V ~ {phi.max():.3f} V")
    print(f"강재 평균 전위: {phi[steel_idx].mean():.3f} V")
    print(f"강재 전류 밀도 범위: {q[steel_idx].min():.3f} A/m² ~ {q[steel_idx].max():.3f} A/m²")

    leg_currents: Dict[int, float] = {}
    for idx in steel_idx:
        leg = elements[idx].leg_index or 0
        leg_currents.setdefault(leg, 0.0)
        leg_currents[leg] += q[idx] * elements[idx].area

    for leg, current in leg_currents.items():
        print(f" - Leg {leg} 순 전류 (강재 유입): {current:.3f} A")

    if anode_idx:
        total_anode_current = sum(q[idx] * elements[idx].area for idx in anode_idx)
        print(f"총 양극 공급 전류: {total_anode_current:.3f} A")


def estimate_anode_consumption(
    elements: Sequence[BoundaryElement],
    result: SimulationResult,
    config: SimulationConfig,
) -> List[Dict[str, float]]:
    """양극 전류를 이용해 질량/두께 감소와 예상 수명을 추정한다."""

    anode_idx = [i for i, el in enumerate(elements) if el.kind == "anode"]
    if not anode_idx:
        return []

    hours = config.service_days * 24.0
    records: List[Dict[str, float]] = []
    for idx in anode_idx:
        element = elements[idx]
        area = element.area
        flux = result.fluxes[idx]
        current = flux * area
        current_abs = abs(current)

        volume = area * config.anode_thickness
        gross_mass = volume * config.anode_density
        usable_mass = gross_mass * config.anode_utilization

        amp_hours = current_abs * hours
        mass_loss = amp_hours / config.anode_capacity if config.anode_capacity > 0 else 0.0

        thickness_loss = (
            mass_loss / (config.anode_density * area)
            if config.anode_density > 0 and area > 0
            else 0.0
        )

        consumed_fraction = (
            min(1.0, mass_loss / usable_mass) if usable_mass > 0 else 0.0
        )

        if current_abs < 1e-12:
            life_days = math.inf
        else:
            usable_ah = usable_mass * config.anode_capacity
            life_hours = usable_ah / current_abs if usable_ah > 0 else math.inf
            life_days = life_hours / 24.0 if life_hours != math.inf else math.inf

        records.append(
            {
                "tag": element.tag or f"anode_{idx}",
                "current": current,
                "mass_loss": mass_loss,
                "thickness_loss": thickness_loss,
                "consumed_fraction": consumed_fraction,
                "life_days": life_days,
            }
        )
    return records


def report_anode_consumption(
    elements: Sequence[BoundaryElement],
    result: SimulationResult,
    config: SimulationConfig,
) -> None:
    """양극 부식 예측 결과를 콘솔에 출력한다."""

    records = estimate_anode_consumption(elements, result, config)
    if not records:
        return

    print("\n=== 양극 부식/소모 예측 ===")
    print(
        f"가정: 서비스 {config.service_days:.0f}일, 용량 {config.anode_capacity:.0f} Ah/kg, "
        f"밀도 {config.anode_density:.0f} kg/m³, 활용률 {config.anode_utilization*100:.1f}%"
    )
    for rec in records:
        life_text = "∞" if math.isinf(rec["life_days"]) else f"{rec['life_days']:.1f}일"
        print(
            f" - {rec['tag']}: 전류 {rec['current']:.3f} A, 질량 감소 {rec['mass_loss']:.3f} kg, "
            f"두께 감소 {rec['thickness_loss']*1e3:.2f} mm, 소비율 {rec['consumed_fraction']*100:.1f}%, "
            f"예상 수명 {life_text}"
        )


def initialize_anode_states(
    elements: Sequence[BoundaryElement], config: SimulationConfig
) -> Dict[int, AnodeLifeState]:
    """동적 해석을 위해 각 양극의 초기 질량 상태를 계산한다."""

    states: Dict[int, AnodeLifeState] = {}
    for idx, element in enumerate(elements):
        if element.kind != "anode":
            continue
        gross_mass = element.area * config.anode_thickness * config.anode_density
        usable_mass = gross_mass * config.anode_utilization
        states[idx] = AnodeLifeState(
            element_index=idx,
            tag=element.tag or f"anode_{idx}",
            initial_area=element.area,
            initial_thickness=config.anode_thickness,
            density=config.anode_density,
            usable_mass=usable_mass,
            remaining_mass=usable_mass,
            current_area=element.area,
            is_active=usable_mass > 0,
        )
    return states


def collect_anode_snapshot(states: Dict[int, AnodeLifeState]) -> List[Dict[str, float]]:
    """현재 양극 상태를 출력용 요약 리스트로 반환한다."""

    snapshot: List[Dict[str, float]] = []
    for state in states.values():
        snapshot.append(
            {
                "tag": state.tag,
                "remaining_mass": state.remaining_mass,
                "usable_mass": state.usable_mass,
                "thickness": state.thickness,
                "area": state.current_area,
                "consumed_fraction": state.consumed_fraction,
                "is_active": state.is_active,
            }
        )
    snapshot.sort(key=lambda rec: rec["tag"])
    return snapshot


def update_anode_states(
    elements: Sequence[BoundaryElement],
    result: SimulationResult,
    config: SimulationConfig,
    states: Dict[int, AnodeLifeState],
    step_days: float,
) -> List[str]:
    """한 스텝 동안의 전류를 이용해 양극 잔량을 갱신하고, 소진된 양극 태그를 반환한다."""

    depleted: List[str] = []
    step_hours = max(step_days, 0.0) * 24.0
    for idx, state in states.items():
        if not state.is_active:
            continue

        element = elements[idx]
        if element.kind != "anode":
            state.is_active = False
            continue

        area = max(element.area, 0.0)
        current_density = result.fluxes[idx]
        current = abs(current_density * area)
        if current <= 0.0 or config.anode_capacity <= 0.0:
            continue

        mass_loss = current * step_hours / config.anode_capacity
        state.remaining_mass = max(0.0, state.remaining_mass - mass_loss)

        if state.remaining_mass <= 0.0:
            element.area = 0.0
            element.kind = "insulator"
            element.fixed_potential = None
            state.is_active = False
            depleted.append(state.tag)
            continue

        state.shrink_geometry()
        element.area = max(state.current_area, 0.0)

        if state.thickness <= config.minimum_anode_thickness or element.area <= 0.0:
            element.area = 0.0
            element.kind = "insulator"
            element.fixed_potential = None
            state.is_active = False
            depleted.append(state.tag)
    return depleted


def report_dynamic_history(history: Sequence[DynamicStepRecord]) -> None:
    """동적 해석 시간 이력을 출력한다."""

    if not history:
        print("동적 해석 결과가 없습니다.")
        return

    print("\n=== 동적 해석 이력 ===")
    for step in history:
        start = step.day
        end = step.day + step.step_days
        phi = step.result.potentials
        active = sum(1 for rec in step.snapshot if rec["is_active"])
        print(
            f"[{start:.1f}~{end:.1f}일] 활성 양극 {active}개, "
            f"전위 {phi.min():.3f}~{phi.max():.3f} V"
        )
        for rec in step.snapshot:
            status = "ON" if rec["is_active"] else "OFF"
            print(
                f"   · {rec['tag']}: 두께 {rec['thickness']*1e3:.2f} mm, "
                f"면적 {rec['area']:.4f} m², "
                f"소비율 {rec['consumed_fraction']*100:.1f}%, 상태 {status}"
            )


def run_dynamic_time_integration(
    config: SimulationConfig,
) -> Tuple[List[BoundaryElement], List[DynamicStepRecord]]:
    """
    지정한 기간(service_days) 동안 시간 스텝별로 전위를 재해석하며
    양극 소모에 따라 경계 조건을 갱신한다.
    """

    elements, _ = build_platform_elements(config)
    polarization = PolarizationModel(tafel_a=config.tafel_a, tafel_b=config.tafel_b)
    states = initialize_anode_states(elements, config)

    history: List[DynamicStepRecord] = []
    current_day = 0.0

    while current_day < config.service_days - 1e-9:
        remaining_days = config.service_days - current_day
        nominal_step = config.dynamic_time_step_days if config.dynamic_time_step_days > 0 else remaining_days
        step_days = min(nominal_step, remaining_days)
        if step_days <= 0:
            break

        solver = CathodicProtectionSolver(elements, polarization)
        result = solver.solve(
            max_iter=config.solver_max_iter,
            tol=config.solver_tol,
            phi_init=config.phi_init,
            q_init=config.q_init,
        )

        depleted = update_anode_states(elements, result, config, states, step_days)
        if depleted:
            print(f"소진된 양극: {', '.join(depleted)}")

        snapshot = collect_anode_snapshot(states)
        history.append(
            DynamicStepRecord(
                day=current_day,
                step_days=step_days,
                result=result,
                snapshot=snapshot,
            )
        )

        current_day += step_days

        if config.stop_when_anodes_depleted:
            still_active = any(state.is_active for state in states.values())
            if not still_active:
                print("모든 양극이 비활성화되어 동적 해석을 조기 종료합니다.")
                break

    return elements, history


def _scatter3d(ax, elements, values=None, cmap="viridis", title=""):
    centers = np.array([el.center for el in elements])
    if values is None:
        colors = [el.kind for el in elements]
        unique = sorted(set(colors))
        color_map = {"steel": "#1f77b4", "insulator": "#7f7f7f", "anode": "#d62728"}
        for kind in unique:
            idx = [i for i, k in enumerate(colors) if k == kind]
            pts = centers[idx]
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=60, label=kind, color=color_map.get(kind, "#333333"))
        ax.legend()
    else:
        sc = ax.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            c=values,
            cmap=cmap,
            s=70,
        )
        plt.colorbar(sc, ax=ax, shrink=0.6, label="값")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))


def plot_results(elements: Sequence[BoundaryElement], result: SimulationResult) -> None:
    centers = np.array([el.center for el in elements])
    phi = result.potentials
    q = result.fluxes

    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(221, projection="3d")
    _scatter3d(ax, elements, title="재킷 플랫폼 메쉬 및 패널")

    ax2 = fig.add_subplot(222, projection="3d")
    _scatter3d(ax2, elements, values=phi, cmap="viridis", title="전위 분포 [V]")

    ax3 = fig.add_subplot(223, projection="3d")
    _scatter3d(ax3, elements, values=q, cmap="plasma", title="전류 밀도 분포 [A/m²]")

    radial = np.linalg.norm(centers[:, :2], axis=1)
    order = np.argsort(radial)
    ax4 = fig.add_subplot(224)
    ax4.plot(radial[order], phi[order], "-o", label="전위 [V]")
    ax4.set_xlabel("중심으로부터 거리 [m]")
    ax4.set_ylabel("전위 [V]")
    ax4.grid(True, linestyle="--", alpha=0.5)

    ax4b = ax4.twinx()
    ax4b.plot(radial[order], q[order], "s--", color="#d62728", label="전류 밀도 [A/m²]")
    ax4b.set_ylabel("전류 밀도 [A/m²]")

    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc="upper right")
    ax4.set_title("거리별 전위/전류 분포 (코너 전류 집중 확인)")

    fig.suptitle("희생 양극 음극 보호 해석 결과", fontsize=15)
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------
# 실행 진입점
# --------------------------------------------------------------------------------------

def run_simulation(config: Optional[SimulationConfig] = None) -> SimulationResult:
    """
    구성 파라미터에 따라 시뮬레이션을 실행하고 결과를 반환한다.
    config가 None이면 기본값을 사용한다.
    """

    if config is None:
        config = SimulationConfig()

    if config.dynamic_mode:
        elements, history = run_dynamic_time_integration(config)
        if not history:
            raise RuntimeError("동적 해석 결과가 생성되지 않았습니다. 입력 파라미터를 확인하세요.")
        final_result = history[-1].result
        summarize_result(elements, final_result)
        report_dynamic_history(history)
        plot_results(elements, final_result)
        return final_result

    elements, _ = build_platform_elements(config)
    polarization = PolarizationModel(tafel_a=config.tafel_a, tafel_b=config.tafel_b)
    solver = CathodicProtectionSolver(elements, polarization)
    result = solver.solve(
        max_iter=config.solver_max_iter,
        tol=config.solver_tol,
        phi_init=config.phi_init,
        q_init=config.q_init,
    )

    summarize_result(elements, result)
    report_anode_consumption(elements, result, config)
    plot_results(elements, result)
    return result


def parse_cli_args() -> SimulationConfig:
    """커맨드라인 인자를 SimulationConfig로 변환한다."""

    parser = argparse.ArgumentParser(
        description="BEM 기반 음극 보호 해석 및 양극 소모 예측"
    )
    parser.add_argument("--base-span", type=float, help="하부 스팬 [m]")
    parser.add_argument("--top-span", type=float, help="상부 스팬 [m]")
    parser.add_argument("--height", type=float, help="플랫폼 높이 [m]")
    parser.add_argument("--num-layers", type=int, help="수직 방향 레이어 개수")
    parser.add_argument("--leg-radius", type=float, help="다리 반경 [m]")
    parser.add_argument("--coating-fraction", type=float, help="상부 절연 비율 (0~1)")
    parser.add_argument("--anode-area", type=float, help="양극 노출 면적 [m²]")
    parser.add_argument("--anode-potential", type=float, help="양극 고정 전위 [V]")
    parser.add_argument("--anode-thickness", type=float, help="양극 두께 [m]")
    parser.add_argument("--anode-density", type=float, help="양극 밀도 [kg/m³]")
    parser.add_argument("--anode-capacity", type=float, help="양극 전기화학 용량 [Ah/kg]")
    parser.add_argument("--anode-utilization", type=float, help="양극 활용률 (0~1)")
    parser.add_argument("--service-days", type=float, help="예상 서비스 기간 [일]")
    parser.add_argument("--solver-tol", type=float, help="솔버 수렴 허용오차")
    parser.add_argument("--solver-max-iter", type=int, help="솔버 최대 반복 횟수")
    parser.add_argument("--phi-init", type=float, help="초기 전위 추정값 [V]")
    parser.add_argument("--q-init", type=float, help="초기 전류밀도 추정값 [A/m²]")
    parser.add_argument("--tafel-a", type=float, help="타펠 기울기 a [V/dec]")
    parser.add_argument("--tafel-b", type=float, help="타펠 절편 b [V]")
    parser.add_argument(
        "--anode-heights",
        type=float,
        nargs="+",
        help="양극 부착 높이 비율(0~1) 목록",
    )
    parser.add_argument(
        "--anode-legs",
        type=int,
        nargs="+",
        help="양극을 부착할 다리 번호 목록 (0~3)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="시간 의존 양극 소모를 고려한 동적 해석 실행",
    )
    parser.add_argument(
        "--time-step-days",
        type=float,
        help="동적 해석 시간 스텝 [일]",
    )
    parser.add_argument(
        "--min-anode-thickness",
        type=float,
        help="양극이 비활성화되기 전 허용 최소 두께 [m]",
    )
    parser.add_argument(
        "--keep-running-without-anode",
        action="store_true",
        help="모든 양극이 소진되어도 계산을 계속 수행",
    )

    args = parser.parse_args()
    config = SimulationConfig()

    for field in [
        "base_span",
        "top_span",
        "height",
        "num_layers",
        "leg_radius",
        "coating_fraction",
        "anode_area",
        "anode_potential",
        "anode_thickness",
        "anode_density",
        "anode_capacity",
        "anode_utilization",
        "service_days",
        "solver_tol",
        "solver_max_iter",
        "phi_init",
        "q_init",
        "tafel_a",
        "tafel_b",
    ]:
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)

    if args.anode_heights:
        config.anode_height_fractions = tuple(args.anode_heights)

    if args.anode_legs:
        config.anode_legs = tuple(args.anode_legs)

    if args.dynamic:
        config.dynamic_mode = True

    if args.time_step_days is not None:
        config.dynamic_time_step_days = args.time_step_days

    if args.min_anode_thickness is not None:
        config.minimum_anode_thickness = args.min_anode_thickness

    if args.keep_running_without_anode:
        config.stop_when_anodes_depleted = False

    return config


if __name__ == "__main__":
    cli_config = parse_cli_args()
    run_simulation(cli_config)

