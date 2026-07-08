import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================================================================
# ส่วนที่ 1: ฟังก์ชันเป้าหมาย (Target Functions)
# จาก pg_engine.js (targets.sin, targets.quad)
# =============================================================================

def target_sin(x):
    """
    ฟังก์ชันเป้าหมาย: f(x) = sin(pi * x)
    รับค่า:
        x (numpy.ndarray): อาร์เรย์ของจุดข้อมูล
    คืนค่า:
        (numpy.ndarray): ผลลัพธ์ของฟังก์ชัน sin(pi * x)
    หลักการคำนวณ:
        แปลง input x ผ่านฟังก์ชัน sine ที่มีความถี่ pi เพื่อสร้างเส้นโค้ง sine wave
    """
    return np.sin(np.pi * x)

def target_quad(x):
    """
    ฟังก์ชันเป้าหมาย: f(x) = x^2
    รับค่า:
        x (numpy.ndarray): อาร์เรย์ของจุดข้อมูล
    คืนค่า:
        (numpy.ndarray): ผลลัพธ์ของฟังก์ชัน x^2
    หลักการคำนวณ:
        ยกกำลังสองของ input x เพื่อสร้างเส้นโค้งพาราโบลา
    """
    return x ** 2

# =============================================================================
# ส่วนที่ 2: การจำลองข้อมูล (Data Simulation)
# จาก pg_engine.js (scenario.sample)
# =============================================================================

def sample_data(target_func, n_samples, add_noise=False, sigma=0.1, seed=None):
    """
    สุ่มข้อมูลจากฟังก์ชันเป้าหมาย (Data Generation)
    รับค่า:
        target_func (callable): ฟังก์ชันเป้าหมาย (เช่น target_sin)
        n_samples (int): จำนวนข้อมูลที่ต้องการสุ่ม
        add_noise (bool): กำหนดว่าจะเพิ่ม Gaussian noise หรือไม่
        sigma (float): ค่าเบี่ยงเบนมาตรฐาน (Standard Deviation) ของ noise
        seed (int): กำหนดค่า random seed เพื่อให้สามารถทดลองซ้ำได้
    คืนค่า:
        x (numpy.ndarray): ข้อมูลแกน x ที่สุ่มมา
        y (numpy.ndarray): ข้อมูลแกน y ที่ได้จากฟังก์ชันเป้าหมายบวก noise
    หลักการคำนวณ:
        1. สุ่มค่า x แบบ Uniform Distribution ในช่วง [-1, 1]
        2. นำ x ไปผ่านฟังก์ชันเป้าหมาย y = f(x)
        3. ถ้า add_noise เป็น True จะบวกค่าที่สุ่มจาก Gaussian Distribution ที่มี mean=0, std=sigma เข้าไปใน y
    """
    rng = np.random.default_rng(seed)
    # สุ่ม x ในช่วง [-1, 1]
    x = rng.uniform(-1, 1, n_samples)
    y = target_func(x)
    
    if add_noise:
        # สุ่ม Gaussian noise
        noise = rng.normal(0, sigma, n_samples)
        y += noise
        
    return x, y

# =============================================================================
# ส่วนที่ 3: แบบจำลอง (Models)
# จาก pg_engine.js (fit.poly)
# =============================================================================

def fit_constant_model(x, y):
    """
    สร้าง Constant Model: h(x) = b
    รับค่า:
        x (numpy.ndarray): ข้อมูลแกน x สำหรับฝึกสอน
        y (numpy.ndarray): ข้อมูลแกน y สำหรับฝึกสอน
    คืนค่า:
        model (function): ฟังก์ชันที่รับค่า x แล้วคืนค่าที่โมเดลทำนาย
    หลักการคำนวณ:
        โมเดลค่าคงที่จะทำนายโดยใช้ค่าเฉลี่ยของ y ทั้งหมด เพื่อทำให้ Mean Squared Error ต่ำที่สุด
    """
    b = np.mean(y)
    return lambda x_new: np.full_like(x_new, b)

def fit_linear_model(x, y):
    """
    สร้าง Linear Model: h(x) = ax + b
    รับค่า:
        x (numpy.ndarray): ข้อมูลแกน x สำหรับฝึกสอน
        y (numpy.ndarray): ข้อมูลแกน y สำหรับฝึกสอน
    คืนค่า:
        model (function): ฟังก์ชันที่รับค่า x แล้วคืนค่าที่โมเดลทำนาย
    หลักการคำนวณ:
        ใช้ scikit-learn LinearRegression ซึ่งใช้วิธี Ordinary Least Squares ในการหาความชัน (a) และจุดตัดแกน (b)
    """
    # ปรับรูปร่างของ x ให้เป็น 2D array สำหรับ scikit-learn
    X = x.reshape(-1, 1)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    
    # ส่งกลับฟังก์ชันเพื่อใช้ทำนาย
    return lambda x_new: lr.predict(x_new.reshape(-1, 1))

def fit_linear_origin_model(x, y):
    """
    สร้าง Linear Model Through Origin: h(x) = ax (ไม่มีจุดตัดแกน)
    รับค่า:
        x (numpy.ndarray): ข้อมูลแกน x สำหรับฝึกสอน
        y (numpy.ndarray): ข้อมูลแกน y สำหรับฝึกสอน
    คืนค่า:
        model (function): ฟังก์ชันที่รับค่า x แล้วคืนค่าที่โมเดลทำนาย
    หลักการคำนวณ:
        ใช้สมการ Normal Equation สำหรับ 1 ตัวแปร: a = (X^T * Y) / (X^T * X)
    """
    # X^T * X
    denominator = np.dot(x, x)
    if denominator == 0:
        a = 0
    else:
        # X^T * Y
        numerator = np.dot(x, y)
        a = numerator / denominator
        
    return lambda x_new: a * x_new

# =============================================================================
# ส่วนที่ 4: การคำนวณ Bias และ Variance (Monte Carlo Simulation)
# ย้ายมาจากส่วนที่หา average fit และ computeRef ใน pg_engine.js
# =============================================================================

def monte_carlo_simulation(target_func, model_type, n_samples, mc_iterations, add_noise=False, sigma=0.1):
    """
    ทำการจำลอง Monte Carlo เพื่อหาความคาดหวังของฟังก์ชันทำนาย (g_bar), Bias^2 และ Variance
    รับค่า:
        target_func (callable): ฟังก์ชันเป้าหมาย
        model_type (str): ประเภทของโมเดล ('constant', 'linear', 'linear_origin')
        n_samples (int): จำนวนจุดข้อมูลในแต่ละชุด (Dataset size)
        mc_iterations (int): จำนวนรอบของการสุ่ม Dataset
        add_noise (bool): เพิ่ม noise หรือไม่
        sigma (float): ค่าเบี่ยงเบนมาตรฐานของ noise
    คืนค่า:
        results (dict): พจนานุกรมเก็บผลลัพธ์ bias^2, variance, e_out, g_bar และรายละเอียดที่ประเมินบนจุดทดสอบต่างๆ
    หลักการคำนวณ:
        1. กำหนดจุดทดสอบ (test points) แกน x สร้างเป็นเส้นประเมินค่าตลอดช่วง [-1, 1]
        2. วนลูปสร้าง Dataset ขนาด n_samples เป็นจำนวน mc_iterations รอบ
        3. ในแต่ละรอบ ฝึกสอนแบบจำลองแล้วทำนายผลลัพธ์บนจุดทดสอบ เก็บผลลัพธ์ทั้งหมดไว้
        4. หาค่า g_bar(x) (ค่าเฉลี่ยของคำทำนาย) ที่แต่ละจุด
        5. Bias^2 คือ ค่าคาดหวังของ (g_bar(x) - f(x))^2 บนจุดทั้งหมด
        6. Variance คือ ค่าคาดหวังของ (g(x) - g_bar(x))^2 บนจุดทั้งหมด
        7. E_out = Bias^2 + Variance + Noise Variance (ถ้ามี)
    """
    # จุดประเมินค่า 100 จุดตั้งฉาก [-1, 1]
    x_test = np.linspace(-1, 1, 100)
    f_x = target_func(x_test)
    
    # เมทริกซ์เก็บผลการทำนายในแต่ละรอบ (รูปทรง: mc_iterations x 100)
    predictions = np.zeros((mc_iterations, len(x_test)))
    
    for i in range(mc_iterations):
        # สุ่มชุดข้อมูลตาม iteration
        x_train, y_train = sample_data(target_func, n_samples, add_noise, sigma, seed=i)
        
        # เลือกแบบจำลองที่ใช้
        if model_type == 'constant':
            model = fit_constant_model(x_train, y_train)
        elif model_type == 'linear':
            model = fit_linear_model(x_train, y_train)
        elif model_type == 'linear_origin':
            model = fit_linear_origin_model(x_train, y_train)
        else:
            raise ValueError("Unknown model type")
            
        # ทำนายผลบนจุดทดสอบ
        predictions[i, :] = model(x_test)
        
    # คำนวณค่าคาดหวังของแบบจำลอง (average model) g_bar(x)
    g_bar = np.mean(predictions, axis=0)
    
    # คำนวณ Bias(x)^2 = (g_bar(x) - f(x))^2
    bias_sq_x = (g_bar - f_x) ** 2
    # ค่า Bias^2 เฉลี่ยตลอดช่วง
    bias_sq = np.mean(bias_sq_x)
    
    # คำนวณ Variance(x) = E[(g(x) - g_bar(x))^2]
    variance_x = np.mean((predictions - g_bar) ** 2, axis=0)
    # ค่า Variance เฉลี่ยตลอดช่วง
    variance = np.mean(variance_x)
    
    # ค่า Noise Variance
    noise_var = (sigma ** 2) if add_noise else 0.0
    
    # E_out = Bias^2 + Variance + Noise
    e_out = bias_sq + variance + noise_var
    
    return {
        'x_test': x_test,
        'f_x': f_x,
        'g_bar': g_bar,
        'bias_sq_x': bias_sq_x,
        'variance_x': variance_x,
        'bias_sq': bias_sq,
        'variance': variance,
        'e_out': e_out,
        'predictions': predictions
    }

# =============================================================================
# ส่วนที่ 5: Learning Curve
# =============================================================================

def generate_learning_curve(target_func, model_type, max_n, mc_iterations, add_noise=False, sigma=0.1):
    """
    สร้างข้อมูลสำหรับ Learning Curve (กราฟ E_out และ E_in เทียบกับขนาดข้อมูล N)
    รับค่า:
        target_func, model_type: ประเภทฟังก์ชันและแบบจำลอง
        max_n (int): จำนวนข้อมูล N สูงสุดที่จะทดสอบ
        mc_iterations (int): จำนวนรอบเฉลี่ย E_in, E_out ในแต่ละค่า N
    คืนค่า:
        n_values, e_ins, e_outs: ลิสต์ของค่า N, ความผิดพลาดตอนฝึก (E_in), ความผิดพลาดจริง (E_out)
    หลักการคำนวณ:
        สำหรับ N แต่ละค่า (ตั้งแต่ 2 ถึง max_n):
        สุ่มข้อมูล N จุดและทดสอบ MC หา E_in กับ E_out เฉลี่ย
    """
    n_values = range(2, max_n + 1)
    e_ins = []
    e_outs = []
    
    # จุดทดสอบสำหรับ E_out
    x_test = np.linspace(-1, 1, 100)
    f_x = target_func(x_test)
    noise_var = (sigma ** 2) if add_noise else 0.0
    
    for n in n_values:
        ein_sum = 0
        eout_sum = 0
        
        for i in range(mc_iterations):
            x_train, y_train = sample_data(target_func, n, add_noise, sigma, seed=i*1000 + n)
            
            if model_type == 'constant':
                model = fit_constant_model(x_train, y_train)
            elif model_type == 'linear':
                model = fit_linear_model(x_train, y_train)
            elif model_type == 'linear_origin':
                model = fit_linear_origin_model(x_train, y_train)
                
            # E_in = MSE on training data
            y_pred = model(x_train)
            ein_sum += np.mean((y_train - y_pred)**2)
            
            # E_out = MSE on test distribution
            test_pred = model(x_test)
            eout_sum += np.mean((test_pred - f_x)**2) + noise_var
            
        e_ins.append(ein_sum / mc_iterations)
        e_outs.append(eout_sum / mc_iterations)
        
    return list(n_values), e_ins, e_outs

# =============================================================================
# ส่วนที่ 6: การแสดงผลและกราฟ (Plotting)
# =============================================================================

def plot_bias_variance(results, model_name):
    """
    วาดกราฟแสดง Target, Average Model และความผันผวน (Variance) รวมถึงการแจกแจง Bias^2
    """
    x = results['x_test']
    
    plt.figure(figsize=(12, 5))
    
    # รูปซ้าย: Target Function, Average Fit, and Individual Fits (Variance)
    plt.subplot(1, 2, 1)
    # วาด predictions 50 เส้นแรกเพื่อให้เห็นความผันผวนของ Variance
    num_lines = min(50, len(results['predictions']))
    for i in range(num_lines):
        plt.plot(x, results['predictions'][i], color='lightblue', alpha=0.1)
    
    plt.plot(x, results['f_x'], 'g-', linewidth=2.5, label='Target f(x)')
    plt.plot(x, results['g_bar'], 'r--', linewidth=2.5, label='Average Fit $\overline{g}(x)$')
    
    plt.title(f"{model_name}\nTarget vs $\overline{{g}}(x)$ (Variance represented by light lines)")
    plt.ylim(-2, 2)
    plt.legend()
    
    # รูปขวา: Bias^2(x) และ Variance(x)
    plt.subplot(1, 2, 2)
    plt.plot(x, results['bias_sq_x'], color='red', linewidth=2, label='Bias$^2(x)$')
    plt.plot(x, results['variance_x'], color='blue', linewidth=2, label='Variance$(x)$')
    
    # เส้นประแนวนอนแสดงค่าเฉลี่ยทั้งหมด
    plt.axhline(results['bias_sq'], color='red', linestyle='--', label=f"Avg Bias$^2$: {results['bias_sq']:.4f}")
    plt.axhline(results['variance'], color='blue', linestyle='--', label=f"Avg Variance: {results['variance']:.4f}")
    
    plt.title(f"Bias$^2$ and Variance distribution over x")
    plt.ylim(0, max(np.max(results['bias_sq_x']), np.max(results['variance_x'])) * 1.2 + 0.1)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_learning_curve(n_values, e_ins, e_outs, model_name):
    """
    วาดกราฟ Learning Curve: N vs Error (E_in, E_out)
    """
    plt.figure(figsize=(8, 5))
    plt.plot(n_values, e_ins, 'b-o', label='$E_{in}$ (Training Error)')
    plt.plot(n_values, e_outs, 'r-o', label='$E_{out}$ (Expected Error)')
    
    plt.title(f"Learning Curve for {model_name}")
    plt.xlabel('Number of Training Examples (N)')
    plt.ylabel('Expected Error')
    plt.ylim(0, max(e_outs) * 1.1 + 0.1)
    plt.legend()
    plt.grid(True)
    plt.show()

def print_comparison_table(models_results):
    """
    พิมพ์ตารางเปรียบเทียบ Bias, Variance, E_out ออกทางหน้าจอ
    """
    print("\n" + "="*65)
    print(f"{'Model':<20} | {'Bias^2':<10} | {'Variance':<10} | {'E_out':<10}")
    print("-" * 65)
    for name, res in models_results.items():
        print(f"{name:<20} | {res['bias_sq']:<10.4f} | {res['variance']:<10.4f} | {res['e_out']:<10.4f}")
    print("="*65 + "\n")
