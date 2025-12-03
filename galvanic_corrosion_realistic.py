"""
==============================================================================
해양 환경에서 희생 양극(Sacrificial Anode)의 갈바닉 부식 시뮬레이션
- 실제 해양 데이터(수온, 염도, 조위) 반영 버전
==============================================================================

개선 사항:
1. 실제 해양 데이터 사용 (수온, 염도, 조위)
2. 온도-염도 기반 전도도 계산
3. 조위에 따른 침수 영역 변화
4. 온도에 따른 부식 속도 변화 (아레니우스 방정식)

저자: AI Assistant
날짜: 2025
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# 폰트 설정 (영어 사용으로 호환성 확보)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 해양 데이터 로드 및 처리
# ==============================================================================

class OceanData:
    """실제 해양 환경 데이터를 로드하고 관리하는 클래스"""
    
    def __init__(self, salinity_file, temperature_file, tidal_file):
        """
        해양 데이터 파일들을 로드합니다.
        
        Args:
            salinity_file: 염도 데이터 CSV 파일 경로
            temperature_file: 수온 데이터 CSV 파일 경로
            tidal_file: 조위 데이터 CSV 파일 경로
        """
        print("Loading ocean data...")
        
        # 데이터 로드
        self.salinity_df = pd.read_csv(salinity_file, parse_dates=['timestamp'])
        self.temperature_df = pd.read_csv(temperature_file, parse_dates=['timestamp'])
        self.tidal_df = pd.read_csv(tidal_file, parse_dates=['timestamp'])
        
        # 데이터 병합
        self.data = self.salinity_df.merge(
            self.temperature_df, on='timestamp'
        ).merge(
            self.tidal_df, on='timestamp'
        )
        
        # 결측치 처리
        self.data = self.data.dropna()
        
        print(f"  - Total data points: {len(self.data):,}")
        print(f"  - Period: {self.data['timestamp'].min()} ~ {self.data['timestamp'].max()}")
        print(f"  - Salinity range: {self.data['salinity'].min():.2f} ~ {self.data['salinity'].max():.2f} psu")
        print(f"  - Temperature range: {self.data['temperature'].min():.2f} ~ {self.data['temperature'].max():.2f} C")
        print(f"  - Tidal level range: {self.data['tidal_level'].min():.2f} ~ {self.data['tidal_level'].max():.2f} cm")
        
    def get_data_at_step(self, step, step_hours=1):
        """
        특정 시뮬레이션 스텝에서의 해양 데이터를 반환합니다.
        
        Args:
            step: 시뮬레이션 스텝 번호
            step_hours: 스텝당 시간 (시간 단위)
            
        Returns:
            dict: 염도, 수온, 조위 데이터
        """
        # 데이터 인덱스 계산 (순환 사용)
        idx = (step * step_hours) % len(self.data)
        
        row = self.data.iloc[idx]
        return {
            'timestamp': row['timestamp'],
            'salinity': row['salinity'],      # psu
            'temperature': row['temperature'], # °C
            'tidal_level': row['tidal_level']  # cm
        }
    
    def calculate_conductivity(self, salinity, temperature):
        """
        염도와 수온으로부터 해수 전도도를 계산합니다.
        
        해수 전도도 경험식 (UNESCO, 1983 기반 간략화):
        σ = C × S × (1 + α(T - T_ref))
        
        여기서:
        - C: 전도도 계수 (~0.12 S/m per psu at 25°C)
        - S: 염도 [psu]
        - α: 온도 계수 (~0.02 /°C)
        - T_ref: 기준 온도 (25°C)
        
        Args:
            salinity: 염도 [psu]
            temperature: 수온 [°C]
            
        Returns:
            conductivity: 전도도 [S/m]
        """
        # 기본 전도도 계수
        C = 0.126  # S/m per psu at 25°C
        T_ref = 25.0  # 기준 온도
        alpha = 0.02  # 온도 계수
        
        # 온도 보정된 전도도
        conductivity = C * salinity * (1 + alpha * (temperature - T_ref))
        
        # 물리적 범위 제한 (3~6 S/m)
        conductivity = np.clip(conductivity, 3.0, 6.0)
        
        return conductivity

# ==============================================================================
# 2. 시뮬레이션 파라미터
# ==============================================================================

class SimulationParameters:
    """시뮬레이션에 사용되는 모든 물리적 파라미터를 관리하는 클래스"""
    
    def __init__(self):
        # ----- 그리드 파라미터 -----
        self.nx = 60  # x 방향 그리드 수
        self.ny = 80  # y 방향 그리드 수 (조위 반영을 위해 높이 증가)
        self.dx = 0.01  # 그리드 간격 [m] (1cm)
        self.dy = 0.01  # 그리드 간격 [m]
        
        # ----- 기본 물리적 파라미터 -----
        self.sigma_base = 4.5  # 기본 해수 전도도 [S/m]
        # phi_anode, phi_cathode는 재료에 따라 자동 설정됨 (아래 참조)
        
        # ----- 아노드 재료 선택 -----
        # 사용 가능한 재료: 'zinc', 'magnesium', 'aluminum'
        self.anode_material = 'zinc'  # ← 여기서 재료 변경!
        
        # 재료별 물성치 데이터베이스
        self.material_properties = {
            'zinc': {
                'name': 'Zinc (Zn)',
                'M': 65.38e-3,      # 몰질량 [kg/mol]
                'z': 2,             # 전자 수 (Zn → Zn²⁺ + 2e⁻)
                'rho': 7140,        # 밀도 [kg/m³]
                'phi': -1.0,        # 전위 [V vs SCE]
                'use': 'Seawater'   # 권장 환경
            },
            'magnesium': {
                'name': 'Magnesium (Mg)',
                'M': 24.31e-3,      # 몰질량 [kg/mol]
                'z': 2,             # 전자 수 (Mg → Mg²⁺ + 2e⁻)
                'rho': 1738,        # 밀도 [kg/m³]
                'phi': -1.5,        # 전위 [V vs SCE]
                'use': 'Freshwater/Soil'
            },
            'aluminum': {
                'name': 'Aluminum (Al)',
                'M': 26.98e-3,      # 몰질량 [kg/mol]
                'z': 3,             # 전자 수 (Al → Al³⁺ + 3e⁻)
                'rho': 2700,        # 밀도 [kg/m³]
                'phi': -1.1,        # 전위 [V vs SCE]
                'use': 'Seawater (High efficiency)'
            }
        }
        
        # 선택된 아노드 재료의 물성치 적용
        mat = self.material_properties[self.anode_material]
        self.M_anode = mat['M']       # 몰질량 [kg/mol]
        self.z_anode = mat['z']       # 전자 수
        self.rho_anode = mat['rho']   # 밀도 [kg/m³]
        self.phi_anode = mat['phi']   # 아노드 전위 [V]
        self.anode_material_name = mat['name']
        
        # ----- 캐소드(선체) 재료 선택 -----
        # 사용 가능한 재료: 'steel', 'stainless_steel', 'copper', 'bronze'
        self.cathode_material = 'steel'  # ← 여기서 재료 변경!
        
        # 캐소드 재료별 전위 데이터베이스
        self.cathode_properties = {
            'steel': {
                'name': 'Carbon Steel (Fe)',
                'phi': -0.6,        # 전위 [V vs SCE]
                'use': 'Ship hull, Pipelines'
            },
            'stainless_steel': {
                'name': 'Stainless Steel',
                'phi': -0.05,       # 전위 [V vs SCE] (더 귀함)
                'use': 'Marine equipment'
            },
            'copper': {
                'name': 'Copper (Cu)',
                'phi': -0.2,        # 전위 [V vs SCE]
                'use': 'Heat exchangers'
            },
            'bronze': {
                'name': 'Bronze (Cu-Sn)',
                'phi': -0.25,       # 전위 [V vs SCE]
                'use': 'Propellers, Valves'
            },
            'cast_iron': {
                'name': 'Cast Iron',
                'phi': -0.5,        # 전위 [V vs SCE]
                'use': 'Pipes, Engine blocks'
            }
        }
        
        # 선택된 캐소드 재료의 전위 적용
        cat = self.cathode_properties[self.cathode_material]
        self.phi_cathode = cat['phi']
        self.cathode_material_name = cat['name']
        
        # 패러데이 상수 (물리 상수)
        self.F = 96485  # [C/mol]
        
        # ----- 온도 의존성 파라미터 (아레니우스 방정식) -----
        self.Ea = 40000  # 활성화 에너지 [J/mol] (부식 반응)
        self.R = 8.314  # 기체 상수 [J/(mol·K)]
        self.T_ref = 298.15  # 기준 온도 [K] (25°C)
        
        # ----- 조위 파라미터 -----
        # 아노드(y=5~35)가 항상 물에 잠기도록 수위 설정
        self.water_level_base = 55  # 기본 수위 (그리드 인덱스) - 아노드 상단(35)보다 높음
        self.tidal_amplitude = 10  # 조위 변화 폭 (그리드 인덱스)
        self.tidal_level_ref = 60  # 조위 기준값 [cm] (데이터 평균)
        self.tidal_scale = 0.15  # 조위 스케일링 계수 (수위 변동 범위 축소)
        
        # ----- 시뮬레이션 파라미터 -----
        self.n_timesteps = 400  # 총 시간 스텝 수
        self.hours_per_step = 6  # 스텝당 시간 [시간]
        self.dt_real = 3600 * self.hours_per_step  # 실제 시간 스텝 [s]
        self.acceleration_factor = 80  # 시간 가속 계수 (더 천천히 부식)
        self.dt = self.dt_real * self.acceleration_factor
        
        # ----- 수치 해석 파라미터 -----
        self.max_iter = 3000  # 라플라스 방정식 최대 반복
        self.tolerance = 1e-5  # 수렴 오차
        self.omega = 1.75  # SOR 파라미터
        
        # ----- 부식 임계값 -----
        self.erosion_threshold = 1.0
        
        # ----- 아노드 위치 설정 (쉽게 조절 가능) -----
        # 아노드 X 위치 (그리드 인덱스)
        self.anode_x_start = 8
        self.anode_x_end = 18
        # 아노드 Y 위치 (수심) - 값이 작을수록 해저에 가까움
        # y=0: 해저, y가 클수록 수면에 가까움
        self.anode_y_start = 5    # 아노드 하단 (해저에서 5cm)
        self.anode_y_end = 35     # 아노드 상단 (해저에서 35cm)
        # 즉, 아노드 높이 = 30 그리드 = 30cm
        
        # ----- 캐소드(선체) 위치 설정 -----
        self.cathode_x_start = 45
        self.cathode_x_end = 55
        self.cathode_y_start = 5   # 선체 하단
        self.cathode_y_end = 75    # 선체 상단 (수면 위까지)

# ==============================================================================
# 3. 도메인 초기화 및 동적 업데이트
# ==============================================================================

def initialize_domain(params):
    """
    시뮬레이션 도메인을 초기화합니다.
    
    도메인 구성:
    - 0: 공기 (물 위)
    - 1: 해수 (Electrolyte)
    - 2: 아노드 (Sacrificial Anode)
    - 3: 캐소드/선체 (Cathode/Hull)
    
    배치 구조 (측면도):
    ─────────────────────── 수면 (y=50~65, 조위에 따라 변동)
           해수
        ┌──┐        ┌────┐
        │아│        │선체│
        │노│        │    │
        │드│        │    │
        └──┘        └────┘
    ─────────────────────── 해저 (y=0)
    
    Returns:
        domain_base: 기본 도메인 (구조물만)
        phi: 초기 전위 분포
        erosion_state: 부식 상태
    """
    domain_base = np.zeros((params.ny, params.nx), dtype=int)
    
    # ----- 아노드 배치 (파라미터에서 위치 가져옴) -----
    # 아노드는 해저 바닥 근처에 배치하여 조위와 관계없이 항상 해수에 잠김
    domain_base[params.anode_y_start:params.anode_y_end, 
                params.anode_x_start:params.anode_x_end] = 2
    
    # ----- 캐소드(선체) 배치 (파라미터에서 위치 가져옴) -----
    # 선체는 해저부터 수면 위까지 뻗어있음
    domain_base[params.cathode_y_start:params.cathode_y_end,
                params.cathode_x_start:params.cathode_x_end] = 3
    
    # 위치 정보 출력
    print(f"Anode position: X=[{params.anode_x_start}-{params.anode_x_end}], "
          f"Y=[{params.anode_y_start}-{params.anode_y_end}] (depth: {params.anode_y_start}-{params.anode_y_end} cm from seabed)")
    print(f"Cathode position: X=[{params.cathode_x_start}-{params.cathode_x_end}], "
          f"Y=[{params.cathode_y_start}-{params.cathode_y_end}]")
    
    # ----- 초기 전위 분포 -----
    phi = np.zeros((params.ny, params.nx))
    
    # ----- 부식 상태 배열 -----
    erosion_state = np.zeros((params.ny, params.nx))
    
    return domain_base, phi, erosion_state

def update_domain_with_tidal(domain_base, tidal_level, params):
    """
    조위에 따라 해수 영역을 업데이트합니다.
    
    Args:
        domain_base: 기본 도메인 (구조물만)
        tidal_level: 현재 조위 [cm]
        params: 시뮬레이션 파라미터
        
    Returns:
        domain: 해수 영역이 업데이트된 도메인
        water_level_idx: 현재 수면 그리드 인덱스
    """
    domain = domain_base.copy()
    
    # 조위를 그리드 인덱스로 변환
    tidal_deviation = (tidal_level - params.tidal_level_ref) * params.tidal_scale
    water_level_idx = int(params.water_level_base + tidal_deviation)
    water_level_idx = np.clip(water_level_idx, 5, params.ny - 5)
    
    # 해수 영역 설정 (수면 아래)
    for i in range(water_level_idx):
        for j in range(params.nx):
            if domain[i, j] == 0:  # 빈 공간만
                domain[i, j] = 1  # 해수로 설정
    
    return domain, water_level_idx

def apply_boundary_conditions(phi, domain, params):
    """전극 경계 조건을 적용합니다."""
    # 아노드: 해수와 접촉한 부분만 활성
    anode_in_water = (domain == 2)
    phi[anode_in_water] = params.phi_anode
    
    # 캐소드: 해수와 접촉한 부분만 활성
    cathode_in_water = (domain == 3)
    phi[cathode_in_water] = params.phi_cathode
    
    return phi

# ==============================================================================
# 4. 온도 의존성 계산
# ==============================================================================

def calculate_temperature_factor(temperature, params):
    """
    아레니우스 방정식을 사용하여 온도에 따른 부식 속도 계수를 계산합니다.
    
    아레니우스 방정식:
    k(T) = A × exp(-Ea / RT)
    
    비율로 표현:
    k(T) / k(T_ref) = exp[(-Ea/R) × (1/T - 1/T_ref)]
    
    Args:
        temperature: 현재 온도 [°C]
        params: 시뮬레이션 파라미터
        
    Returns:
        factor: 온도 보정 계수 (기준 온도에서 1.0)
    """
    T = temperature + 273.15  # 켈빈으로 변환
    T_ref = params.T_ref
    
    # 아레니우스 비율
    factor = np.exp((-params.Ea / params.R) * (1/T - 1/T_ref))
    
    # 범위 제한 (0.5 ~ 2.0)
    factor = np.clip(factor, 0.5, 2.0)
    
    return factor

# ==============================================================================
# 5. 라플라스 방정식 풀이 (벡터화 + SOR)
# ==============================================================================

def solve_laplace_fdm(phi, domain, params, sigma):
    """
    Solves Laplace equation using FDM + SOR method.
    
    Key physics:
    - Seawater (domain=1): Laplace equation applies
    - Anode (domain=2): Dirichlet BC (fixed potential)
    - Cathode (domain=3): Dirichlet BC (fixed potential)
    - Air (domain=0): Insulator - Neumann BC (no current flux)
    
    Water surface treatment:
    - Air-water interface is an INSULATOR (not ground!)
    - Apply Neumann BC: ∂φ/∂n = 0 at water surface
    - Implementation: When a seawater cell neighbors air, use the seawater 
      cell's own value instead of air's value (mirror boundary condition)
    """
    phi = phi.copy()
    omega = params.omega
    
    # Seawater mask
    seawater_mask = (domain == 1)
    
    for iteration in range(params.max_iter):
        phi_old = phi.copy()
        
        # Iterate over seawater cells only
        for i in range(1, params.ny - 1):
            for j in range(1, params.nx - 1):
                if domain[i, j] != 1:  # Skip if not seawater
                    continue
                
                # Calculate neighbor average with proper boundary handling
                neighbors_sum = 0.0
                n_neighbors = 0
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < params.ny and 0 <= nj < params.nx:
                        neighbor_domain = domain[ni, nj]
                        
                        if neighbor_domain == 0:
                            # AIR (insulator): Apply Neumann BC
                            # Use current cell's value (mirror condition)
                            # This ensures ∂φ/∂n = 0 at water surface
                            neighbors_sum += phi[i, j]
                        else:
                            # Seawater, Anode, or Cathode: use neighbor's value
                            neighbors_sum += phi[ni, nj]
                        
                        n_neighbors += 1
                
                if n_neighbors > 0:
                    phi_gs = neighbors_sum / n_neighbors
                    phi[i, j] = phi[i, j] + omega * (phi_gs - phi[i, j])
        
        # Domain boundary conditions (outer walls - insulated)
        phi[0, :] = phi[1, :]
        phi[-1, :] = phi[-2, :]
        phi[:, 0] = phi[:, 1]
        phi[:, -1] = phi[:, -2]
        
        # Re-apply electrode potentials (Dirichlet BC)
        phi = apply_boundary_conditions(phi, domain, params)
        
        # Convergence check
        max_change = np.max(np.abs(phi[seawater_mask] - phi_old[seawater_mask]))
        if max_change < params.tolerance:
            return phi, True
    
    return phi, False

# ==============================================================================
# 6. 전류 밀도 계산
# ==============================================================================

def calculate_current_density(phi, domain, params, sigma):
    """전류 밀도를 계산합니다."""
    grad_phi_x = np.zeros_like(phi)
    grad_phi_y = np.zeros_like(phi)
    
    # 중심 차분
    grad_phi_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * params.dx)
    grad_phi_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * params.dy)
    
    # 경계
    grad_phi_x[:, 0] = (phi[:, 1] - phi[:, 0]) / params.dx
    grad_phi_x[:, -1] = (phi[:, -1] - phi[:, -2]) / params.dx
    grad_phi_y[0, :] = (phi[1, :] - phi[0, :]) / params.dy
    grad_phi_y[-1, :] = (phi[-1, :] - phi[-2, :]) / params.dy
    
    jx = -sigma * grad_phi_x
    jy = -sigma * grad_phi_y
    j_magnitude = np.sqrt(jx**2 + jy**2)
    
    return jx, jy, j_magnitude

def calculate_anode_surface_current(domain, j_magnitude, params):
    """아노드 표면 전류를 계산합니다."""
    # 해수와 접하는 아노드 표면 검출
    anode_mask = (domain == 2)
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    
    seawater_mask = (domain == 1)
    seawater_dilated = ndimage.binary_dilation(seawater_mask, structure)
    
    surface_mask = anode_mask & seawater_dilated
    
    # 표면 전류 계산
    surface_current = np.zeros_like(j_magnitude)
    
    for i in range(params.ny):
        for j in range(params.nx):
            if surface_mask[i, j]:
                neighbors_j = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < params.ny and 0 <= nj < params.nx:
                        if domain[ni, nj] == 1:
                            neighbors_j.append(j_magnitude[ni, nj])
                
                if neighbors_j:
                    surface_current[i, j] = np.mean(neighbors_j)
    
    # 총 전류량
    pixel_area = params.dx * params.dy
    total_current = np.sum(surface_current[surface_mask]) * pixel_area
    active_surface_area = np.sum(surface_mask) * pixel_area
    
    return surface_current, surface_mask, total_current, active_surface_area

# ==============================================================================
# 7. 아노드 형상 업데이트
# ==============================================================================

def update_anode_geometry(domain_base, erosion_state, surface_current, 
                          surface_mask, params, temp_factor):
    """패러데이 법칙으로 아노드 형상을 업데이트합니다."""
    domain_base = domain_base.copy()
    erosion_state = erosion_state.copy()
    
    # 부식 속도 (온도 보정 적용)
    erosion_rate_factor = (params.M_anode * params.dt * temp_factor) / \
                          (params.z_anode * params.F * params.rho_anode)
    
    mass_loss = 0.0
    pixel_volume = params.dx * params.dy * params.dx
    
    for i in range(params.ny):
        for j in range(params.nx):
            if surface_mask[i, j] and domain_base[i, j] == 2:
                j_local = surface_current[i, j]
                erosion_depth = erosion_rate_factor * j_local
                erosion_increment = erosion_depth / params.dx
                erosion_state[i, j] += erosion_increment
                
                if erosion_state[i, j] >= params.erosion_threshold:
                    remaining = 1.0 - (erosion_state[i, j] - erosion_increment)
                    mass_loss += pixel_volume * params.rho_anode * max(0, remaining)
                    domain_base[i, j] = 0  # 빈 공간으로
                    erosion_state[i, j] = 0.0
    
    return domain_base, erosion_state, mass_loss

# ==============================================================================
# 8. 메인 시뮬레이션 루프
# ==============================================================================

def run_simulation(params, ocean_data):
    """Main simulation loop."""
    print("=" * 70)
    print("Galvanic Corrosion Simulation (with Real Ocean Data)")
    print("=" * 70)
    print(f"Grid size: {params.nx} x {params.ny}")
    print(f"Time steps: {params.n_timesteps} x {params.hours_per_step} hours")
    print(f"Acceleration factor: {params.acceleration_factor}")
    print("-" * 70)
    print(f"Anode Material: {params.anode_material_name}")
    print(f"  - Molar mass: {params.M_anode*1000:.2f} g/mol")
    print(f"  - Density: {params.rho_anode} kg/m³")
    print(f"  - Electrons: {params.z_anode}")
    print(f"  - Potential: {params.phi_anode} V")
    print(f"Cathode Material: {params.cathode_material_name}")
    print(f"  - Potential: {params.phi_cathode} V")
    print(f"Galvanic Driving Voltage: {abs(params.phi_anode - params.phi_cathode):.2f} V")
    print("=" * 70)
    
    # 초기화
    domain_base, phi, erosion_state = initialize_domain(params)
    initial_anode_pixels = np.sum(domain_base == 2)
    
    # 결과 저장
    history = {
        'domains': [],
        'phis': [],
        'total_currents': [],
        'anode_areas': [],
        'mass_losses': [],
        'time_steps': [],
        'temperatures': [],
        'salinities': [],
        'conductivities': [],
        'tidal_levels': [],
        'water_levels': [],
        'temp_factors': []
    }
    
    cumulative_mass_loss = 0.0
    
    for step in range(params.n_timesteps):
        # ----- 해양 데이터 가져오기 -----
        ocean = ocean_data.get_data_at_step(step, params.hours_per_step)
        temperature = ocean['temperature']
        salinity = ocean['salinity']
        tidal_level = ocean['tidal_level']
        
        # ----- 환경 파라미터 계산 -----
        sigma = ocean_data.calculate_conductivity(salinity, temperature)
        temp_factor = calculate_temperature_factor(temperature, params)
        
        # ----- 조위에 따른 도메인 업데이트 -----
        domain, water_level_idx = update_domain_with_tidal(
            domain_base, tidal_level, params
        )
        
        # ----- 전위 초기화 및 경계조건 -----
        phi[domain == 1] = (params.phi_anode + params.phi_cathode) / 2
        phi = apply_boundary_conditions(phi, domain, params)
        
        # ----- 라플라스 방정식 풀이 -----
        phi, converged = solve_laplace_fdm(phi, domain, params, sigma)
        
        # ----- 전류 밀도 계산 -----
        jx, jy, j_magnitude = calculate_current_density(phi, domain, params, sigma)
        
        # ----- 표면 전류 계산 -----
        surface_current, surface_mask, total_current, active_area = \
            calculate_anode_surface_current(domain, j_magnitude, params)
        
        # ----- 형상 업데이트 -----
        domain_base, erosion_state, mass_loss = update_anode_geometry(
            domain_base, erosion_state, surface_current, 
            surface_mask, params, temp_factor
        )
        
        cumulative_mass_loss += mass_loss
        
        # ----- 결과 기록 -----
        current_anode_pixels = np.sum(domain_base == 2)
        time_days = (step + 1) * params.hours_per_step / 24
        
        history['domains'].append(domain.copy())
        history['phis'].append(phi.copy())
        history['total_currents'].append(total_current)
        history['anode_areas'].append(active_area)
        history['mass_losses'].append(cumulative_mass_loss)
        history['time_steps'].append(time_days)
        history['temperatures'].append(temperature)
        history['salinities'].append(salinity)
        history['conductivities'].append(sigma)
        history['tidal_levels'].append(tidal_level)
        history['water_levels'].append(water_level_idx)
        history['temp_factors'].append(temp_factor)
        
        # ----- Progress output -----
        if step % 30 == 0 or step == params.n_timesteps - 1:
            anode_ratio = current_anode_pixels / initial_anode_pixels * 100
            print(f"Step {step:3d}/{params.n_timesteps}: "
                  f"Current={total_current*1000:.2f}mA, "
                  f"Anode={anode_ratio:.1f}%, "
                  f"T={temperature:.1f}C, "
                  f"S={salinity:.1f}psu, "
                  f"sigma={sigma:.2f}S/m, "
                  f"Tide={tidal_level:.0f}cm")
        
        # Check if anode is consumed
        if current_anode_pixels == 0:
            print("\nAnode completely consumed!")
            break
    
    print("=" * 70)
    print("Simulation Complete!")
    final_ratio = current_anode_pixels / initial_anode_pixels * 100
    print(f"Final anode: {final_ratio:.1f}%")
    print(f"Total mass loss: {cumulative_mass_loss*1000:.2f} g")
    print("=" * 70)
    
    return history

# ==============================================================================
# 9. 시각화 함수들
# ==============================================================================

def create_animation(history, params, save_path='galvanic_corrosion_realistic.gif'):
    """Creates corrosion process animation."""
    # Colormap: Air, Seawater, Anode, Cathode
    colors = ['#E3F2FD', '#1565C0', '#FF8C00', '#616161']
    cmap = ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle('Galvanic Corrosion Simulation\n(with Real Ocean Data)', 
                 fontsize=14, fontweight='bold')
    
    legend_patches = [
        mpatches.Patch(color='#E3F2FD', label='Air'),
        mpatches.Patch(color='#1565C0', label='Seawater'),
        mpatches.Patch(color='#FF8C00', label='Anode'),
        mpatches.Patch(color='#616161', label='Hull (Cathode)')
    ]
    
    def animate(frame):
        for ax in axes:
            ax.clear()
        
        domain = history['domains'][frame]
        phi = history['phis'][frame]
        
        # Domain shape
        axes[0].imshow(domain, cmap=cmap, vmin=0, vmax=3, origin='lower',
                       extent=[0, params.nx*params.dx*100, 0, params.ny*params.dy*100])
        axes[0].axhline(y=history['water_levels'][frame]*params.dy*100, 
                       color='cyan', linestyle='--', linewidth=2, label='Water Surface')
        axes[0].set_xlabel('X [cm]', fontsize=11)
        axes[0].set_ylabel('Y [cm]', fontsize=11)
        axes[0].set_title(f'Domain (Step {frame})', fontsize=12)
        axes[0].legend(handles=legend_patches, loc='upper right', fontsize=8)
        
        # Potential distribution
        im = axes[1].imshow(phi, cmap='RdYlBu_r', origin='lower',
                           extent=[0, params.nx*params.dx*100, 0, params.ny*params.dy*100],
                           vmin=params.phi_anode, vmax=params.phi_cathode)
        axes[1].set_xlabel('X [cm]', fontsize=11)
        axes[1].set_ylabel('Y [cm]', fontsize=11)
        axes[1].set_title('Potential Distribution [V]', fontsize=12)
        
        # Environment info
        info_text = (f'Time: {history["time_steps"][frame]:.1f} days\n'
                    f'Temp: {history["temperatures"][frame]:.1f} C\n'
                    f'Salinity: {history["salinities"][frame]:.1f} psu\n'
                    f'Conductivity: {history["conductivities"][frame]:.2f} S/m\n'
                    f'Tidal Level: {history["tidal_levels"][frame]:.0f} cm')
        axes[1].text(0.02, 0.98, info_text, transform=axes[1].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return axes
    
    n_frames = min(len(history['domains']), 150)  # Max 150 frames
    frame_indices = np.linspace(0, len(history['domains'])-1, n_frames, dtype=int)
    
    anim = FuncAnimation(fig, animate, frames=frame_indices, interval=100, repeat=True)
    
    print(f"Saving animation: {save_path}")
    anim.save(save_path, writer='pillow', fps=10, dpi=100)
    print("Animation saved!")
    
    plt.close()
    return anim

def plot_comprehensive_results(history, params, save_path='galvanic_results_realistic.png'):
    """Creates comprehensive result graphs."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('Galvanic Corrosion Simulation Results (with Real Ocean Data)', 
                 fontsize=14, fontweight='bold')
    
    time_days = history['time_steps']
    
    # 1. Current change
    ax1 = axes[0, 0]
    currents_mA = [c * 1000 for c in history['total_currents']]
    ax1.plot(time_days, currents_mA, 'b-', linewidth=1.5, alpha=0.7)
    ax1.fill_between(time_days, currents_mA, alpha=0.3)
    ax1.set_xlabel('Time [days]')
    ax1.set_ylabel('Total Current [mA]')
    ax1.set_title('Total Current vs Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. Anode area change
    ax2 = axes[0, 1]
    initial_area = history['anode_areas'][0] if history['anode_areas'][0] > 0 else 1
    area_ratio = [a / initial_area * 100 for a in history['anode_areas']]
    ax2.plot(time_days, area_ratio, 'g-', linewidth=2)
    ax2.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% lifetime')
    ax2.set_xlabel('Time [days]')
    ax2.set_ylabel('Active Surface Area [%]')
    ax2.set_title('Anode Active Surface Area')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Temperature and conductivity
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    l1, = ax3.plot(time_days, history['temperatures'], 'r-', linewidth=1, 
                   alpha=0.7, label='Temperature')
    l2, = ax3_twin.plot(time_days, history['conductivities'], 'b-', 
                        linewidth=1, alpha=0.7, label='Conductivity')
    ax3.set_xlabel('Time [days]')
    ax3.set_ylabel('Temperature [C]', color='red')
    ax3_twin.set_ylabel('Conductivity [S/m]', color='blue')
    ax3.set_title('Temperature and Seawater Conductivity')
    ax3.legend(handles=[l1, l2], loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. Salinity and tidal level
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    l1, = ax4.plot(time_days, history['salinities'], 'm-', linewidth=1, 
                   alpha=0.7, label='Salinity')
    l2, = ax4_twin.plot(time_days, history['tidal_levels'], 'c-', 
                        linewidth=1, alpha=0.7, label='Tidal Level')
    ax4.set_xlabel('Time [days]')
    ax4.set_ylabel('Salinity [psu]', color='purple')
    ax4_twin.set_ylabel('Tidal Level [cm]', color='cyan')
    ax4.set_title('Salinity and Tidal Level')
    ax4.legend(handles=[l1, l2], loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # 5. Temperature correction factor
    ax5 = axes[2, 0]
    ax5.plot(time_days, history['temp_factors'], 'orange', linewidth=1.5)
    ax5.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time [days]')
    ax5.set_ylabel('Temperature Correction Factor')
    ax5.set_title('Corrosion Rate Temperature Factor\n(Arrhenius Equation)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Current-environment correlation
    ax6 = axes[2, 1]
    scatter = ax6.scatter(history['conductivities'], currents_mA, 
                         c=history['temperatures'], cmap='coolwarm',
                         alpha=0.6, s=20)
    ax6.set_xlabel('Conductivity [S/m]')
    ax6.set_ylabel('Total Current [mA]')
    ax6.set_title('Conductivity vs Current (color: Temperature)')
    plt.colorbar(scatter, ax=ax6, label='Temperature [C]')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Results saved: {save_path}")
    plt.close()

def plot_current_vs_time_detailed(history, save_path='current_vs_time_detailed.png'):
    """Creates detailed current vs time graph."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_days = history['time_steps']
    currents_mA = [c * 1000 for c in history['total_currents']]
    
    # Current plot
    ax.plot(time_days, currents_mA, 'b-', linewidth=2, marker='', label='Total Current')
    ax.fill_between(time_days, currents_mA, alpha=0.3, color='blue')
    
    # Moving average
    window = min(20, len(currents_mA) // 5)
    if window > 1:
        moving_avg = np.convolve(currents_mA, np.ones(window)/window, mode='valid')
        time_avg = time_days[window//2:window//2+len(moving_avg)]
        ax.plot(time_avg, moving_avg, 'r-', linewidth=2.5, label=f'Moving Avg ({window} steps)')
    
    ax.set_xlabel('Time [days]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Current [mA]', fontsize=12, fontweight='bold')
    ax.set_title('Total Current vs Time\n(with Real Ocean Data - Temperature/Salinity/Tidal variation)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper right', fontsize=10)
    
    # Statistics
    initial_current = currents_mA[0] if currents_mA else 0
    final_current = currents_mA[-1] if currents_mA else 0
    mean_current = np.mean(currents_mA)
    
    textstr = '\n'.join([
        f'Initial: {initial_current:.2f} mA',
        f'Final: {final_current:.2f} mA',
        f'Mean: {mean_current:.2f} mA',
        f'Reduction: {(1-final_current/initial_current)*100:.1f}%' if initial_current > 0 else ''
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Current-time graph saved: {save_path}")
    plt.close()

# ==============================================================================
# 10. 메인 실행
# ==============================================================================

if __name__ == "__main__":
    # Load ocean data
    try:
        ocean_data = OceanData(
            salinity_file='ocean_data/hourly_avg_water_salinity.csv',
            temperature_file='ocean_data/hourly_avg_water_temperature.csv',
            tidal_file='ocean_data/hourly_avg_water_tidal_level.csv'
        )
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        print("Cannot run simulation without ocean data.")
        ocean_data = None
    
    # Simulation parameters
    params = SimulationParameters()
    
    if ocean_data is not None:
        # Run simulation
        history = run_simulation(params, ocean_data)
        
        # Visualization
        print("\nGenerating visualizations...")
        
        # 1. Animation
        create_animation(history, params, 'galvanic_corrosion_realistic.gif')
        
        # 2. Comprehensive results
        plot_comprehensive_results(history, params, 'galvanic_results_realistic.png')
        
        # 3. Current-time graph
        plot_current_vs_time_detailed(history, 'current_vs_time_detailed.png')
        
        print("\nAll visualizations complete!")
        print("Generated files:")
        print("  - galvanic_corrosion_realistic.gif (animation)")
        print("  - galvanic_results_realistic.png (comprehensive results)")
        print("  - current_vs_time_detailed.png (current vs time graph)")
    else:
        print("Cannot run simulation without ocean data.")

