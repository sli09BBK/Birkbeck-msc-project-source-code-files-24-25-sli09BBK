import tkinter as tk
from tkinter import ttk
import os
from tkinter import messagebox
from DrissionPage import ChromiumPage
import pickle
import time
from DrissionPage.common import Actions
import threading
import csv  # Import the csv module
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.join(os.path.dirname(__file__),'..'))))
from X_Rednote_Kafka.Kafka_producer import XiaohongshuKafkaProducer


class App:
    def __init__(self):
        # Create a user data directory
        """
        Folder Structure:
        data->  goods_name_1 -> links -> link1.txt
                                      -> link2.txt
                             -> notebooks -> notebook1.csv # 修改为 .csv
            ->  goods_name_2
            ->  cookies -> name_1.pkl
                        -> name_2.pkl
        """
        # --- 修正开始：修改数据保存路径到当前项目目录 ---
        # 获取当前脚本文件所在的目录
        current_script_directory = os.path.dirname(os.path.abspath(__file__))
        # 将 'data' 文件夹添加到当前脚本目录下
        data_folder = os.path.join(current_script_directory, "data")
        # --- 修正结束 ---

        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        self.data_folder = data_folder
        self.cookies_folder = os.path.join(self.data_folder, "cookies")
        if not os.path.exists(self.cookies_folder):
            os.makedirs(self.cookies_folder)
        # 小红书的平台url
        self.platform_url = "https://www.xiaohongshu.com"
        # 窗口设置
        self.root = tk.Tk()
        self.root.geometry('300x200')
        self.root.title("Xiaohongshu Crawler Tool")
        self.root.resizable(False, False)
        # 设置菜单
        self.menubar = tk.Menu(self.root)
        self.edit_menu = tk.Menu(self.menubar, tearoff=0)
        # 将命令绑定到启动新线程的方法，避免 GUI 卡死
        self.edit_menu.add_command(label="Add Account Cookies", command=self.start_add_cookies_thread)
        self.edit_menu.add_command(label="Delete Account Cookies", command=self.delete_cookies)
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu)
        self.menubar.add_cascade(label="Instructions", command=self.options)
        self.root.config(menu=self.menubar)
        # 设置爬虫页面
        self.good_name_label = tk.Label(self.root, text="Enter search content below:")
        self.good_name_label.pack()
        self.good_name_entry = tk.Entry(self.root)
        self.good_name_entry.pack()
        self.accounts_label = tk.Label(self.root, text="Select account:")
        self.accounts_label.pack()
        self.accounts_combobox = ttk.Combobox(self.root)
        self.accounts_combobox.pack()

        # 调用新方法来更新下拉列表，确保在程序启动时显示现有账户
        self.update_accounts_combobox()

        # 将命令绑定到启动新线程的方法，避免 GUI 卡死
        self.button = tk.Button(self.root, text="Start Crawling", command=self.start_crawling_thread)
        self.button.pack()
        self.root.mainloop()

    # 添加这个新方法来更新 combobox
    def update_accounts_combobox(self):
        # 确保 cookies 文件夹存在再列出其内容
        if not os.path.exists(self.cookies_folder):
            os.makedirs(self.cookies_folder)
        all_accounts = os.listdir(self.cookies_folder)
        # 过滤出以 .pkl 结尾的文件，只显示 cookie 文件
        all_accounts = [f for f in all_accounts if f.endswith('.pkl')]
        self.accounts_combobox["values"] = all_accounts
        if all_accounts:
            # 如果有账户，默认选中第一个
            self.accounts_combobox.set(all_accounts[0])
        else:
            # 如果没有账户，清空选择
            self.accounts_combobox.set("")

    # 添加一个启动新线程来执行 add_cookies 逻辑的方法
    def start_add_cookies_thread(self):
        # 在新线程中运行 add_cookies 逻辑
        window = tk.Toplevel(self.root)
        window.geometry('300x200')
        window.title("Add Account Cookies")
        # 提示用户不需要输入扩展名
        name_label = tk.Label(window, text="Enter account name (no need for .pkl extension):")
        name_label.pack()
        name_entry = tk.Entry(window)
        name_entry.pack()

        def confirm_and_thread():
            account_name_input = name_entry.get().strip()  # 获取用户输入并去除首尾空格
            if not account_name_input:
                messagebox.showerror(message="Account name cannot be empty!")
                return

            # 清除用户可能输入的 .pkl 扩展名，确保文件名一致性
            if account_name_input.lower().endswith('.pkl'):
                account_name_input = account_name_input[:-4]  # 移除 '.pkl'

            # 禁用按钮防止重复点击
            button.config(state=tk.DISABLED)
            # 在新线程中运行真正的 confirm 逻辑
            thread = threading.Thread(target=self._run_add_cookies_logic, args=(account_name_input, window))
            thread.start()

        button = tk.Button(window, text="Confirm", command=confirm_and_thread)
        button.pack()

    # 真正的 add_cookies 逻辑，将在新线程中运行
    def _run_add_cookies_logic(self, account_name, parent_window):
        messagebox.showinfo(
            message="Please open Xiaohongshu and prepare to scan the QR code. You only have 60 seconds to complete scanning, otherwise the account may not be logged in properly!")
        page = None  # 初始化为 None
        try:
            page = ChromiumPage()
            page.get(self.platform_url)
            page.set.window.max()  # 最大化窗口，方便用户扫码

            # 等待手动身份验证 - 这是阻塞点，现在在线程中
            # 增加循环检查登录状态，而不是固定等待
            login_successful = False
            # 新的 XPath 检查：根据当前小红书登录后的元素
            # 尝试查找"发布"按钮（红框+加号）或者顶部导航栏的头像/用户名区域
            for _ in range(12):  # 尝试检查12次，每次等待5秒，总共60秒
                time.sleep(5)
                # 检查发布按钮（可能在登录后出现）
                # 这是根据你提供的图片猜测的XPath，你可能需要根据实际页面元素调整
                if page.ele('xpath://*[@class="reds-icon icon-add-bold"]') or \
                        page.ele('xpath://*[@id="global"]/div[2]/div[1]/ul/div/li[4]'):  # 假设这是头像或个人中心入口
                    login_successful = True
                    break
                print("Waiting for login...")  # 可以在控制台输出，帮助调试

            if not login_successful:
                self.root.after(0, lambda: messagebox.showerror(message="Login timed out or not detected. Please try again!"))
                return  # 提前返回，不保存cookies

            cookies = page.cookies()
            # 确保这里始终添加 .pkl 扩展名
            file_path = os.path.join(self.cookies_folder, f"{account_name}.pkl")

            with open(file_path, "wb") as cookies_file:
                pickle.dump(cookies, cookies_file)
            self.root.after(0, lambda: messagebox.showinfo(message="Cookies saved successfully!"))  # 在主线程显示
        except Exception as e_info:
            self.root.after(0, lambda info=e_info: messagebox.showerror(
                message=f"Failed to add cookies: {info}\nPlease ensure network connection is stable and QR code scanning is completed."))
        finally:
            if page:
                page.quit()  # 确保浏览器关闭
            # 更新 GUI 元素必须在主线程中完成，这里使用 after
            self.root.after(0, self.update_accounts_combobox)  # 更新下拉列表
            self.root.after(0, parent_window.destroy)  # 关闭添加窗口

    # 添加一个启动新线程来执行 start_crawling 逻辑的方法
    def start_crawling_thread(self):
        # 禁用按钮防止重复点击
        self.button.config(state=tk.DISABLED)

        # 在新线程中运行真正的爬虫逻辑
        thread = threading.Thread(target=self._run_crawling_logic)
        thread.start()

    # 真正的 start_crawling 逻辑，将在新线程中运行
    def _run_crawling_logic(self):
        def remove_character(string):
            result = ""
            for _ in string:
                if _.isdigit():
                    result = result + _
            return result

        # GUI 消息必须在主线程中显示，这里直接调用 messagebox
        messagebox.showinfo(
            message="The code will run for some time. To ensure all notes are crawled successfully, please keep your screen on (or stay at your computer). Thank you for using!")

        goods_name = self.good_name_entry.get()
        if not goods_name:
            self.root.after(0, lambda: messagebox.showerror(message="Search content cannot be empty!"))
            self.root.after(0, lambda: self.button.config(state=tk.NORMAL))  # 重新启用按钮
            return

        if not os.path.exists(os.path.join(self.data_folder, goods_name)):
            os.makedirs(os.path.join(self.data_folder, goods_name))
        goods_folder = os.path.join(self.data_folder, goods_name)
        selected_value = self.accounts_combobox.get()

        if not selected_value:
            self.root.after(0, lambda: messagebox.showerror(message="Please select an account for crawling!"))
            self.root.after(0, lambda: self.button.config(state=tk.NORMAL))  # 重新启用按钮
            return

        page = None  # 初始化为 None
        try:
            # 这里的 selected_value 已经包含 .pkl 扩展名，直接使用即可
            path = os.path.join(self.cookies_folder, selected_value)

            if not os.path.exists(path):
                self.root.after(0, lambda: messagebox.showerror(
                    message=f"Cookie file '{selected_value}' does not exist. Please add or check."))
                self.root.after(0, lambda: self.button.config(state=tk.NORMAL))
                return

            with open(path, "rb") as cookies_file:
                cookies = pickle.load(cookies_file)

            page = ChromiumPage()  # 在线程中创建浏览器实例
            for cookie in cookies:
                page.set.cookies(cookie)
            page.get(self.platform_url)
            page.set.window.max()
            time.sleep(2)

            # 使用新的 XPath 来定位搜索框，根据你提供的图片信息调整
            search_input = page.ele('xpath://*[@id="search-input"]')  # 假设搜索框的ID是search-input
            if search_input:
                search_input.input(goods_name)
                ac = Actions(page)
                ac.key_down("ENTER")
                ac.key_up("ENTER")
            else:
                self.root.after(0, lambda: messagebox.showerror(message="Search input field not found. Please check page structure or network connection."))
                self.root.after(0, lambda: self.button.config(state=tk.NORMAL))
                return

            time.sleep(2)

            try:
                # 修正查找笔记选项卡的方式，根据你提供的图片信息调整
                # 假设搜索结果页面的"笔记"选项卡可以通过ID 'exploreFeeds'下的第一个div来定位
                note_tab = page.ele('xpath://*[@id="exploreFeeds"]/section[1]')
                if note_tab:
                    note_tab.click()
                    time.sleep(1)
                else:
                    print("Notes tab element not found. Page structure may have changed or already on notes page.")
                    # 如果没有找到，尝试直接在搜索结果页面进行滚动
            except Exception as e:
                print(f"Error when trying to click notes tab: {e}")

            all_links = set()
            scroll_count = 0
            max_scrolls = 50  # 设置一个最大滚动次数，防止无限循环
            while True:
                last_num = len(all_links)
                # 根据你提供的图片信息，调整小红书笔记封面元素的XPath
                current_notes = page.eles('xpath://*[@class="cover mask ld"]')  # 假设是这个class
                if current_notes:
                    for _ in current_notes:
                        link = _.attr("href")
                        if link and link.startswith("/"):  # 确保链接是相对路径，并拼接完整URL
                            all_links.add(self.platform_url + link)
                        elif link and link.startswith("http"):  # 如果已经是完整URL
                            all_links.add(link)

                now_num = len(all_links)
                page.actions.scroll(800)
                time.sleep(1)  # 增加等待时间，确保页面加载和内容渲染
                print(f"Number of links: {len(all_links)}, increasing, scrolling down 800 units!")

                scroll_count += 1
                if now_num == last_num or scroll_count >= max_scrolls:
                    # 如果链接数量不再增加，或者达到最大滚动次数，则退出
                    print("Number of links no longer increasing or maximum scroll count reached. Stopping scrolling.")
                    break

            if not all_links:
                self.root.after(0,
                                lambda: messagebox.showwarning(message="No note links were crawled. Please check your search keyword or network."))
                self.root.after(0, lambda: self.button.config(state=tk.NORMAL))
                return

            all_links = list(all_links)
            links_path = os.path.join(goods_folder, "links.txt")
            with open(links_path, "w", encoding="utf-8") as f:
                for i in range(len(all_links)):
                    f.write(all_links[i] + "\n")
                    print(f"Successfully wrote link #{i + 1} to file!")

            # 关闭当前页面，开始处理详细笔记
            if page:
                page.quit()

            if not os.path.exists(os.path.join(goods_folder, "notebooks")):
                os.makedirs(os.path.join(goods_folder, "notebooks"))
            notebooks_folder = os.path.join(goods_folder, "notebooks")

            # --- 修正开始：保存为 CSV 格式 ---
            csv_file_path = os.path.join(notebooks_folder, f"{goods_name}_notes.csv")
            csv_headers = ["Author ID", "Title", "Content", "Publish Time", "Likes", "Favorites", "Comments"]

            # 检查 CSV 文件是否存在，如果不存在则写入表头
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, "w", newline="", encoding="utf-8-sig") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(csv_headers)
            # --- 修正结束 ---

            with open(links_path, "r", encoding="utf-8") as f:
                all_links = [_.rstrip() for _ in f.readlines()]

            # 初始化Kafka Producer
            kafka_producer = XiaohongshuKafkaProducer()

            i = 1
            for link in all_links:
                page_note = None  # 为每篇笔记创建新的 page 实例
                try:
                    print(f"Starting to crawl note #{i}!")
                    # 加载 cookies
                    with open(path, "rb") as cookies_file:  # 这里仍然使用 'path'，它包含了完整的 cookie 文件路径
                        cookies = pickle.load(cookies_file)

                    page_note = ChromiumPage()  # 为每篇笔记创建新的浏览器实例
                    for cookie in cookies:
                        page_note.set.cookies(cookie)

                    page_note.get(link)
                    time.sleep(5)  # 等待页面完全加载

                    # 检查页面是否加载成功，例如通过检查是否存在笔记标题元素
                    if not page_note.ele('@id=detail-title') and not page_note.ele('@class=note-text'):
                        print(f"Link {link} page load failed or content does not exist. Skipping this note.")
                        continue  # 跳过当前笔记

                    # 优化元素查找，增加健壮性
                    title_element = page_note.ele("xpath://*[@id='detail-title']")
                    title = title_element.attr("text") if title_element else "N/A"
                    # 清除标题中的换行符和多余空格
                    title = title.replace('\n', ' ').strip()
                    print(f"Title is {title}")

                    ID_element = page_note.ele("xpath://*[@class='username']")
                    ID = ID_element.attr("text") if ID_element else "N/A"
                    print(f"ID is {ID}")

                    content_element = page_note.ele("xpath://*[@class='note-text']")
                    content = content_element.attr("text") if content_element else "N/A"
                    # 清除内容中的换行符和多余空格
                    content = content.replace('\n', ' ').strip()
                    print(f"Content for title {title}, ID {ID} has been extracted.")

                    date_element = page_note.ele("xpath://*[@class='date']")
                    date = date_element.attr("text") if date_element else "N/A"
                    print(f"Publish time is {date}.")

                    # 重新定位点赞、收藏、评论元素，小红书页面结构变化频繁
                    # 尝试更通用的 XPath 或使用 DrissionPage 的其他选择器
                    # 根据你提供的图片信息，调整XPath
                    likes_element = page_note.ele(
                        'xpath://*[@id="noteContainer"]/div[4]/div[3]/div/div/div[1]/div[2]/div/div[1]/span[1]')
                    likes = likes_element.attr("text") if likes_element else "0"

                    collect_element = page_note.ele(
                        'xpath://*[@id="note-page-collect-board-guide"]')
                    collect = collect_element.attr("text") if collect_element else "0"

                    chat_element = page_note.ele(
                        'xpath://*[@id="noteContainer"]/div[4]/div[3]/div/div/div[1]/div[2]/div/div[1]/span[3]')
                    chat = chat_element.attr("text") if chat_element else "0"

                    print(f"Likes: {likes}, Favorites: {collect}, Comments: {chat}")

                    # --- 修正开始：将数据写入 CSV ---
                    with open(csv_file_path, "a", newline="", encoding="utf-8-sig") as csv_file:
                        csv_writer = csv.writer(csv_file)
                        row_data = [
                            ID,
                            title,
                            content,
                            date,
                            remove_character(likes),
                            remove_character(collect),
                            remove_character(chat)
                        ]
                        csv_writer.writerow(row_data)
                    print(f"Note #{i} has been crawled and written to CSV!")
                    # --- 修正结束 ---
                    # --- Kafka发送开始 ---
                    row_data = {
                        'author': ID,
                        'title': title,
                        'content': content,
                        'publish_time': date,
                        'likes': remove_character(likes),
                        'favorites': remove_character(collect),
                        'comments': remove_character(chat)
                    }
                    kafka_producer.send_data(row_data)
                    print(f"Note #{i} has been crawled and sent to Kafka!")
                    # --- Kafka发送结束 ---

                except Exception as e:
                    print(f"Error occurred while crawling note #{i}: {e}. Skipping this note. Link: {link}")
                finally:
                    if page_note:  # 确保关闭的是当前笔记的浏览器实例
                        page_note.quit()
                    time.sleep(5)  # 暂停 5 秒，降低请求频率
                    i += 1

            # 所有任务完成后，通过 after 在主线程中显示完成消息和启用按钮
            self.root.after(0, lambda: messagebox.showinfo(message="All notes have been crawled! Data has been saved to CSV file!"))
            self.root.after(0, lambda: self.button.config(state=tk.NORMAL))  # 重新启用按钮

        except Exception as e_info:
            # 如果发生任何异常，通过 after 在主线程中显示错误消息和启用按钮
            self.root.after(0, lambda info=e_info: messagebox.showerror(message=f"Error occurred during crawling: {info}"))
            self.root.after(0, lambda: self.button.config(state=tk.NORMAL))
        finally:
            if page:  # 确保关闭最初的浏览器实例
                page.quit()

    # delete_cookies 和 options 方法保持不变，因为它们通常不会阻塞
    def delete_cookies(self):
        def delete():
            selected_value = combobox.get()
            if not selected_value:
                messagebox.showerror(message="Please select an account to delete!")
                return
            file_path = os.path.join(self.cookies_folder, selected_value)
            try:
                os.remove(file_path)
                messagebox.showinfo(message=f"{selected_value} has been deleted!")
                # 删除后，通过 after 在主线程更新 GUI
                self.root.after(0, self.update_accounts_combobox)
                self.root.after(0, window.destroy)
            except FileNotFoundError:
                messagebox.showerror(message=f"File '{selected_value}' not found.")
            except Exception as e:
                messagebox.showerror(message=f"Error occurred while deleting file: {e}")

        window = tk.Toplevel(self.root)
        window.geometry('300x200')
        window.title("Delete Account Cookies")
        label = tk.Label(window, text="Select the cookies to delete below:")
        label.pack()
        combobox = ttk.Combobox(window)
        combobox.pack()

        data = [f for f in os.listdir(self.cookies_folder) if f.endswith('.pkl')]
        combobox["values"] = data
        if data:
            combobox.set(data[0])

        button = tk.Button(window, text="Delete", command=delete)
        button.pack()

    def options(self):
        window = tk.Toplevel(self.root)
        window.geometry("400x200")
        window.resizable(False, False)
        title = "Operation Steps:"
        label = tk.Label(window, text=title)
        label.pack()
        first = tk.Label(window, text="1. First click the 'Add Account Cookies' button in the Edit menu to get cookies.")
        first.pack()
        second = tk.Label(window, text="2. Then enter search content in the input box and select corresponding cookies to start crawling.")
        second.pack()
        final = tk.Label(window, text="3. Finally wait for the program to complete and check the data folder for results.")
        final.pack()
        note = tk.Label(window, text="Note: If you encounter anti-crawling measures, you can switch to another account for the next operation!")
        note.pack()


# 主程序入口
if __name__ == '__main__':
    App()
