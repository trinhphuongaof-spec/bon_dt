import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

# Thay thế st.title bằng st.markdown để tùy chỉnh màu sắc và chữ in hoa
st.markdown("<h1 style='color: blue;'>ỨNG DỤNG PHÂN TÍCH BÁO CÁO TÀI CHÍNH 📊</h1>", unsafe_allow_html=True)

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100
    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")
    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]
    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9
    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df
# --- Hàm gọi API Gemini (Phân tích) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản, khả năng thanh toán (hiện hành và nhanh), hiệu quả hoạt động (lợi nhuận ròng) và đòn bẩy tài chính (nợ trên vốn chủ sở hữu).
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm xử lý Chatbot ---
def chat_with_gemini(prompt, api_key):
    """Xử lý yêu cầu chat với Gemini API, duy trì lịch sử cuộc trò chuyện."""
    try:
        client = genai.Client(api_key=api_key)
        # Khởi tạo Chat Session nếu chưa có
        if 'chat_session' not in st.session_state:
            # System instruction để Gemini tập trung vào chuyên môn tài chính
            system_instruction = (
                "Bạn là một chuyên gia cố vấn tài chính am hiểu về phân tích báo cáo tài chính "
                "và các chỉ số kinh tế. Hãy trả lời các câu hỏi về tài chính, kinh tế, "
                "kế toán một cách chuyên nghiệp và dễ hiểu. Nếu câu hỏi không liên quan, "
                "hãy lịch sự từ chối và yêu cầu hỏi về chủ đề tài chính."
            )
            config = genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
            st.session_state.chat_session = client.chats.create(
                model="gemini-2.5-flash",  # Model thích hợp cho hội thoại
                config=config
            )
        
        # Gửi tin nhắn và nhận phản hồi
        response = st.session_state.chat_session.send_message(prompt)
        return response.text
        
    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong quá trình chat: {e}"

# -------------------------- BẮT ĐẦU GIAO DIỆN STREAMLIT --------------------------

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)
if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())
        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo dict để lưu trữ các chỉ số. Dùng dict để dễ dàng đưa vào AI.
            ratios = {}
            
            try:
                # ------------------ KHAI THÁC DỮ LIỆU TỪ DATAFRAME ------------------
                
                def get_val(keyword, year_col):
                    """Hàm helper để lấy giá trị cho MẪU SỐ, trả về 1e-9 (thay cho 0) nếu không tìm thấy, giúp tránh lỗi chia cho 0."""
                    try:
                        # Lấy giá trị sau khi lọc
                        val = df_processed[df_processed['Chỉ tiêu'].str.contains(keyword, case=False, na=False)][year_col].iloc[0]
                        # Trả về 1e-9 nếu giá trị xấp xỉ 0 để tránh lỗi chia
                        return val if val != 0 else 1e-9 
                    except IndexError:
                        # Nếu không tìm thấy, trả về 1e-9 để tránh lỗi chia
                        return 1e-9 
                
                def get_numerator_val(keyword, year_col):
                    """Hàm helper để lấy giá trị cho TỬ SỐ, trả về 0 nếu không tìm thấy (cho phép LNTT âm/bằng 0)."""
                    try:
                        return df_processed[df_processed['Chỉ tiêu'].str.contains(keyword, case=False, na=False)][year_col].iloc[0]
                    except IndexError:
                        return 0.0

                # Lấy các chỉ tiêu cốt lõi
                # Khả năng Thanh toán
                tsnh_n = get_numerator_val('TÀI SẢN NGẮN HẠN', 'Năm sau')
                tsnh_n_1 = get_numerator_val('TÀI SẢN NGẮN HẠN', 'Năm trước')
                htk_n = get_numerator_val('HÀNG TỒN KHO', 'Năm sau') # Hàng tồn kho
                htk_n_1 = get_numerator_val('HÀNG TỒN KHO', 'Năm trước')
                no_ngan_han_N = get_val('NỢ NGẮN HẠN', 'Năm sau')  # Mẫu số
                no_ngan_han_N_1 = get_val('NỢ NGẮN HẠN', 'Năm trước') # Mẫu số
                
                # Đòn bẩy
                tong_no_n = get_numerator_val('TỔNG CỘNG NỢ PHẢI TRẢ', 'Năm sau') # Tổng Nợ
                tong_no_n_1 = get_numerator_val('TỔNG CỘNG NỢ PHẢI TRẢ', 'Năm trước')
                vcsh_n = get_val('VỐN CHỦ SỞ HỮU', 'Năm sau') # Vốn Chủ Sở Hữu (Mẫu số)
                vcsh_n_1 = get_val('VỐN CHỦ SỞ HỮU', 'Năm trước') # Vốn Chủ Sở Hữu (Mẫu số)
                
                # Hiệu quả hoạt động
                lnst_n = get_numerator_val('LỢI NHUẬN SAU THUẾ', 'Năm sau') # Lợi nhuận sau thuế (Tử số)
                lnst_n_1 = get_numerator_val('LỢI NHUẬN SAU THUẾ', 'Năm trước') # Lợi nhuận sau thuế (Tử số)
                dt_n = get_val('DOANH THU THUẦN', 'Năm sau') # Doanh thu thuần (Mẫu số)
                dt_n_1 = get_val('DOANH THU THUẦN', 'Năm trước') # Doanh thu thuần (Mẫu số)
                
                # ------------------ TÍNH TOÁN CÁC CHỈ SỐ ------------------

                # 1. Chỉ số Thanh toán Hiện hành (Current Ratio)
                ratios['Thanh_toan_HH_N'] = tsnh_n / no_ngan_han_N
                ratios['Thanh_toan_HH_N_1'] = tsnh_n_1 / no_ngan_han_N_1
                
                # 2. Chỉ số Thanh toán Nhanh (Quick Ratio)
                ratios['Thanh_toan_Nhanh_N'] = (tsnh_n - htk_n) / no_ngan_han_N
                ratios['Thanh_toan_Nhanh_N_1'] = (tsnh_n_1 - htk_n_1) / no_ngan_han_N_1
                
                # 3. Tỷ suất Lợi nhuận Ròng (Net Profit Margin) - Tính bằng %
                ratios['Loi_nhuan_Rong_N'] = (lnst_n / dt_n) * 100 
                ratios['Loi_nhuan_Rong_N_1'] = (lnst_n_1 / dt_n_1) * 100

                # 4. Tỷ suất Nợ trên Vốn Chủ Sở Hữu (D/E Ratio)
                ratios['No_tren_VCSH_N'] = tong_no_n / vcsh_n
                ratios['No_tren_VCSH_N_1'] = tong_no_n_1 / vcsh_n_1

                # ------------------ HIỂN THỊ KẾT QUẢ TRÊN STREAMLIT ------------------
                
                def display_ratio(label, ratio_n_1, ratio_n, is_percentage=False):
                    """Hàm helper hiển thị 1 chỉ số và so sánh giữa 2 năm."""
                    # Định dạng chuỗi giá trị và delta
                    try:
                        val_n_1 = f"{ratio_n_1:.2f}{'%' if is_percentage else ' lần'}"
                        val_n = f"{ratio_n:.2f}{'%' if is_percentage else ' lần'}"
                        delta = f"{ratio_n - ratio_n_1:.2f}"
                    except:
                        val_n_1 = "Lỗi tính toán"
                        val_n = "Lỗi tính toán"
                        delta = "N/A"

                    col_prev, col_current = st.columns(2)
                    with col_prev:
                        st.metric(
                            label=f"{label} (Năm trước)",
                            value=val_n_1
                        )
                    with col_current:
                        st.metric(
                            label=f"{label} (Năm sau)",
                            value=val_n,
                            delta=delta
                        )
                
                st.markdown("#### **Khả năng Thanh toán**")
                display_ratio("Chỉ số Thanh toán Hiện hành", ratios['Thanh_toan_HH_N_1'], ratios['Thanh_toan_HH_N'])
                display_ratio("Chỉ số Thanh toán Nhanh (Acid-test)", ratios['Thanh_toan_Nhanh_N_1'], ratios['Thanh_toan_Nhanh_N'])
                
                st.markdown("#### **Hiệu quả hoạt động & Đòn bẩy**")
                display_ratio("Tỷ suất Lợi nhuận Ròng", ratios['Loi_nhuan_Rong_N_1'], ratios['Loi_nhuan_Rong_N'], is_percentage=True)
                display_ratio("Tỷ suất Nợ trên Vốn Chủ Sở Hữu", ratios['No_tren_VCSH_N_1'], ratios['No_tren_VCSH_N'])

            except Exception as e:
                 # Cảnh báo nếu thiếu bất kỳ chỉ tiêu cốt lõi nào
                 st.warning(f"Thiếu chỉ tiêu cần thiết để tính đầy đủ các chỉ số tài chính (Có thể thiếu: TÀI SẢN NGẮN HẠN, HÀNG TỒN KHO, LỢI NHUẬN SAU THUẾ, DOANH THU THUẦN, TỔNG CỘNG NỢ PHẢI TRẢ, VỐN CHỦ SỞ HỮU). Chi tiết lỗi: {e}")
                 # Đảm bảo dict ratios tồn tại và chứa tất cả keys để tránh lỗi ở Chức năng 5
                 ratios = {k: "N/A" for k in [
                     'Thanh_toan_HH_N', 'Thanh_toan_HH_N_1', 
                     'Thanh_toan_Nhanh_N', 'Thanh_toan_Nhanh_N_1', 
                     'Loi_nhuan_Rong_N', 'Loi_nhuan_Rong_N_1', 
                     'No_tren_VCSH_N', 'No_tren_VCSH_N_1'
                 ]}
            
            # --------------------------------------------------------------------------------------
            # --- Chức năng 5: Nhận xét AI ---
            # --------------------------------------------------------------------------------------
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (Sử dụng dict ratios)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán Hiện hành (N-1)', 
                    'Thanh toán Hiện hành (N)',
                    'Thanh toán Nhanh (N-1)',
                    'Thanh toán Nhanh (N)',
                    'Tỷ suất LN Ròng (N-1)',
                    'Tỷ suất LN Ròng (N)',
                    'Nợ trên VCSH (N-1)',
                    'Nợ trên VCSH (N)',
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    # Tăng trưởng tài sản ngắn hạn
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if 'TÀI SẢN NGẮN HẠN' in df_processed['Chỉ tiêu'].str.upper().to_list() else "N/A", 
                    f"{ratios['Thanh_toan_HH_N_1']}", 
                    f"{ratios['Thanh_toan_HH_N']}",
                    f"{ratios['Thanh_toan_Nhanh_N_1']}",
                    f"{ratios['Thanh_toan_Nhanh_N']}",
                    f"{ratios['Loi_nhuan_Rong_N_1']}",
                    f"{ratios['Loi_nhuan_Rong_N']}",
                    f"{ratios['No_tren_VCSH_N_1']}",
                    f"{ratios['No_tren_VCSH_N']}",
                ]
            }).to_markdown(index=False) 
            
            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                     st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# --------------------------------------------------------------------------------------
# --- BỔ SUNG CHỨC NĂNG 6: CHATBOT HỎI ĐÁP TÀI CHÍNH (GEMINI) ---
# --------------------------------------------------------------------------------------

# Khởi tạo lịch sử chat nếu chưa có trong session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("---") # Đường kẻ ngang phân cách
st.subheader("6. Chatbot Hỏi đáp Tài chính (Gemini) 🤖")
st.caption("Hãy hỏi Gemini về các khái niệm, công thức hoặc ý nghĩa của các chỉ số tài chính, ví dụ: 'Tỷ suất lợi nhuận gộp là gì?'")

# Hiển thị lịch sử tin nhắn
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý đầu vào từ người dùng
if prompt := st.chat_input("Hỏi Gemini về Tài chính..."):
    
    # Thêm tin nhắn người dùng vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    api_key = st.secrets.get("GEMINI_API_KEY") 
    if not api_key:
        ai_response = "Lỗi: Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng Chatbot."
    else:
        # Gọi hàm chat
        with st.spinner("Đang chờ Gemini trả lời..."):
            ai_response = chat_with_gemini(prompt, api_key)

    # Thêm tin nhắn AI vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
